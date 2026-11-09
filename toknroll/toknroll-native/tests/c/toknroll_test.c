/* libtoknroll's C API, function by function: every documented result, error code and special case, on
 * the tokenizers fixtures.h writes, so every id is known and nothing is downloaded. Built twice more:
 * against libtoknroll.a (TOKNROLL_STATIC), and with TEST_LIBJINFER against libjinfer, which carries the
 * same functions on its own runtime, set by JINFER_* variables, refusing forked children.
 *
 *   ./toknroll_test           every scenario, each in a fresh process
 *   ./toknroll_test <name>    one, here */
#include "test.h"   /* first: it picks the POSIX features */
#include "fixtures.h"
#include "toknroll.h"

#ifdef TEST_LIBJINFER
#  define PROGRAM "jinfer"
#  define LOG_VAR "JINFER_LOG"
#  define OPTIONS_VAR "JINFER_OPTIONS"
#  define WORKER "Pool-worker-"       /* a batch's threads: libjinfer's run on the common pool */
#else
#  define PROGRAM "toknroll"
#  define LOG_VAR "TOKNROLL_LOG"
#  define OPTIONS_VAR "TOKNROLL_OPTIONS"
#  define WORKER "batch-"             /* libtoknroll's own, toknroll-batch-N */
#endif

#define LE toknroll_last_error
static const int32_t HELLO[] = {ID_HE, 108, 108, 111};
/* the one warning a scenario can count on offline: an option that replaces a registered pre-tokenizer */
#define REPLACE "-Dtoknroll.gguf.pre.gpt2=alias:qwen2"
#define REPLACED REPLACE " replaces the registered pre-tokenizer 'gpt2'"
static const char* const REPLACE_GPT2[] = {REPLACE};
/* what makes the ctest-pre GGUF loadable: its pre-tokenizer, aliased to a known one */
#define ALIAS_CTEST "-Dtoknroll.gguf.pre.ctest-pre=alias:gpt2"

static const char* tiny_json(void) {
    const char* path = TMP("tiny.json");
    write_tiny_json(path);
    return path;
}

static toknroll_tokenizer* tiny(void) {
    toknroll_tokenizer* t = toknroll_tokenizer_load(tiny_json());
    if (!t) { printf("    the fixture does not load: %s\n", LE(NULL)); exit(1); }
    return t;
}

static int same_ids(const int32_t* a, const int32_t* b, int n) { return !memcmp(a, b, (size_t) n * sizeof *a); }

/* ---- version, last error, and what never starts the runtime ---- */

static void version(void) {
    char want[32];
    snprintf(want, sizeof want, "%d.%d.%d", TOKNROLL_VERSION_MAJOR, TOKNROLL_VERSION_MINOR, TOKNROLL_VERSION_PATCH);
    CHECK_STR(toknroll_version(), want);
    CHECK_CLEAR(LE);                     /* before any call: "" and OK */
    CHECK_STR(LE(NULL), "");             /* code may be NULL */
    toknroll_tokenizer_free(NULL);       /* accepted, and starts nothing */
    /* none of the three started the runtime: configure still comes first */
    CHECK_EQ(toknroll_configure(NULL, 0), TOKNROLL_OK);
    CHECK_CLEAR(LE);
}

static void last_error_per_call(void) {
    toknroll_tokenizer* t = tiny();
    CHECK_CLEAR(LE);
    CHECK_ERROR(toknroll_encode(t, "x", 1, 2, NULL, 0), TOKNROLL_EINVAL, LE, "unknown flags");
    CHECK_HAS(LE(NULL), "unknown flags");          /* reading changes nothing */
    CHECK_HAS(LE(NULL), "unknown flags");
    toknroll_version();                            /* nor does toknroll_version */
    CHECK_HAS(LE(NULL), "unknown flags");
    toknroll_tokenizer_free(NULL);                 /* nor a free of NULL */
    CHECK_HAS(LE(NULL), "unknown flags");
    CHECK_EQ(toknroll_vocabulary_size(t), VOCAB_JSON);   /* the next call does */
    CHECK_CLEAR(LE);
    CHECK_ERROR(toknroll_vocabulary_size(NULL), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    toknroll_tokenizer_free(t);
    CHECK_CLEAR(LE);                               /* a free is a call too */
}

typedef struct { toknroll_tokenizer* t; int32_t first_code, own_code; char own[128]; } errors_box;
static void other_thread_errors(void* p) {
    errors_box* box = (errors_box*) p;
    LE(&box->first_code);                          /* nothing called here yet: OK */
    toknroll_token(box->t, -5, NULL, 0);
    const char* m = LE(&box->own_code);
    snprintf(box->own, sizeof box->own, "%s", m);
}

typedef struct { int32_t code; char message[128]; } peeked;
static void peek(void* p) {
    peeked* r = (peeked*) p;
    snprintf(r->message, sizeof r->message, "%s", LE(&r->code));
}

static void start_the_runtime(void* p) {
    (void) p;
    toknroll_tokenizer_free(tiny());
}

/* before the runtime starts, as after, a failure is the failing thread's own */
static void last_error_before_start(void) {
    CHECK_ERROR(toknroll_configure(NULL, 1), TOKNROLL_EINVAL, LE, "options is NULL");
    peeked other = {99, "unread"};
    thread_join(thread_run(peek, &other));
    CHECK_EQ(other.code, TOKNROLL_OK);   /* a thread that never called sees nothing */
    CHECK_STR(other.message, "");
    CHECK(strstr(LE(NULL), "options is NULL") != NULL);   /* the failing thread keeps its own */
    /* and keeps it once another thread starts the runtime, until its own next call */
    thread_join(thread_run(start_the_runtime, NULL));
    CHECK_ERROR(TOKNROLL_EINVAL, TOKNROLL_EINVAL, LE, "options is NULL");
    toknroll_tokenizer* t = tiny();
    CHECK_CLEAR(LE);
    toknroll_tokenizer_free(t);
}

static void last_error_per_thread(void) {
    errors_box box = {tiny(), 99, 99, ""};
    CHECK_ERROR(toknroll_decode(box.t, NULL, -1, 0, NULL, 0, NULL), TOKNROLL_EINVAL, LE, "count must not be negative");
    thread_join(thread_run(other_thread_errors, &box));
    CHECK_EQ(box.first_code, TOKNROLL_OK);
    CHECK_EQ(box.own_code, TOKNROLL_ENOTFOUND);
    CHECK_HAS(box.own, "no token with id -5");
    CHECK_HAS(LE(NULL), "count must not be negative");   /* this thread's own, untouched */
    toknroll_tokenizer_free(box.t);
}

/* vocabulary_size is one past the largest id; an id below it no token has is no token anywhere */
static void vocabulary_with_a_gap(void) {
    const char* path = TMP("gapped.json");
    write_gapped_json(path);
    toknroll_tokenizer* t = toknroll_tokenizer_load(path);
    CHECK(t != NULL);
    if (!t) return;
    CHECK_EQ(toknroll_vocabulary_size(t), 301);
    CHECK_EQ(toknroll_id(t, "he", 2), 300);
    int32_t ids[8];
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 8), 4);
    CHECK_EQ(ids[0], 300);
    char out[16];
    CHECK_EQ(toknroll_decode(t, ids, 4, 0, out, sizeof out, NULL), 5);
    CHECK_STR(out, "hello");
    CHECK_EQ(toknroll_token_flags(t, ID_END), TOKNROLL_TOKEN_CONTROL);   /* the added tokens keep theirs */
    const int32_t gaps[] = {256, 259, 299};
    for (int i = 0; i < COUNT(gaps); i++) {
        CHECK_ERROR(toknroll_token(t, gaps[i], out, sizeof out), TOKNROLL_ENOTFOUND, LE, "no token with id");
        CHECK_ERROR(toknroll_token_flags(t, gaps[i]), TOKNROLL_ENOTFOUND, LE, "no token with id");
        int32_t two[] = {300, gaps[i]}, decoded = -7;
        CHECK_ERROR(toknroll_decode(t, two, 2, 0, out, sizeof out, &decoded), TOKNROLL_EINVAL, LE, ", at ids[1]");
        CHECK_EQ(decoded, -7);   /* left as it was */
    }
    CHECK_ERROR(toknroll_token(t, 301, out, sizeof out), TOKNROLL_ENOTFOUND, LE, "no token with id 301");
    toknroll_tokenizer_free(t);
}

/* ---- configure ---- */

static void configure_arguments(void) {
    const char* one[] = {"-Dtoknroll.offline=true", NULL};
    CHECK_ERROR(toknroll_configure(NULL, 1), TOKNROLL_EINVAL, LE, "toknroll_configure: options is NULL or count negative");
    CHECK_ERROR(toknroll_configure(one, -1), TOKNROLL_EINVAL, LE, "options is NULL or count negative");
    CHECK_ERROR(toknroll_configure(one, 2), TOKNROLL_EINVAL, LE, "toknroll_configure: options[1] is NULL");
    const char* blank[] = {"-Dtoknroll.offline=true", " \t"};
    CHECK_ERROR(toknroll_configure(blank, 2), TOKNROLL_EINVAL, LE, "toknroll_configure: options[1] is blank");
    blank[1] = "";
    CHECK_ERROR(toknroll_configure(blank, 2), TOKNROLL_EINVAL, LE, "options[1] is blank");
    /* a refused call is no call: the next one still configures */
    CHECK_EQ(toknroll_configure(one, 1), TOKNROLL_OK);
    CHECK_CLEAR(LE);
    CHECK_ERROR(toknroll_configure(one, 1), TOKNROLL_EINVAL, LE, "toknroll_configure must come once, before any other call into the library");
    CHECK_ERROR(toknroll_configure(NULL, 0), TOKNROLL_EINVAL, LE, "must come once");
    toknroll_tokenizer* t = tiny();
    CHECK_EQ(toknroll_vocabulary_size(t), VOCAB_JSON);
    toknroll_tokenizer_free(t);
}

static void configure_after_a_call(void) {
    toknroll_tokenizer_free(tiny());
    CHECK_ERROR(toknroll_configure(NULL, 0), TOKNROLL_EINVAL, LE, "toknroll_configure must come once");
}

static void configure_rejected(void) {
    const char* bad[] = {"-XX:+NoSuchOptionAnywhere"};
    capture_stderr();
    CHECK_ERROR(toknroll_configure(bad, 1), TOKNROLL_ERROR, LE, "an option from toknroll_configure was rejected, see stderr for which");
    char* err = captured_stderr();
    CHECK_HAS(err, "NoSuchOptionAnywhere");   /* the runtime names it */
    free(err);
    /* no runtime, for good: every call says why */
    CHECK_NULL_ERROR(toknroll_tokenizer_load(tiny_json()), TOKNROLL_ERROR, LE, "the runtime could not be started");
    CHECK_ERROR(toknroll_vocabulary_size(NULL), TOKNROLL_ERROR, LE, "could not be started");
    CHECK_ERROR(toknroll_configure(NULL, 0), TOKNROLL_EINVAL, LE, "toknroll_configure must come once");
    CHECK_ERROR(toknroll_log_set(NULL, NULL), TOKNROLL_EINVAL, LE, "toknroll_log_set must come once");
    /* that refusal is the last call's; the next call that finds no runtime says why there is none again */
    CHECK_ERROR(toknroll_vocabulary_size(NULL), TOKNROLL_ERROR, LE, "the runtime could not be started");
    CHECK(*toknroll_version());   /* no runtime needed */
}

static void configure_ignores_unknown_x(void) {
    const char* x[] = {"-Xnosuchthing"};
    CHECK_EQ(toknroll_configure(x, 1), TOKNROLL_OK);
    toknroll_tokenizer_free(tiny());
    CHECK_CLEAR(LE);
}

/* ctest-pre is no scheme toknroll knows: only an option makes the GGUF loadable */
static const char* custom_gguf(void) {
    const char* path = TMP("custom.gguf");
    write_tiny_gguf(path, "ctest-pre");
    return path;
}

static void options_unknown_pre(void) {
    CHECK_NULL_ERROR(toknroll_tokenizer_load(custom_gguf()), TOKNROLL_EUNSUPPORTED, LE, "Unsupported GGUF pre-tokenizer key 'ctest-pre'");
}

static void options_take_effect(void) {
    const char* alias[] = {ALIAS_CTEST};
    CHECK_EQ(toknroll_configure(alias, 1), TOKNROLL_OK);
    toknroll_tokenizer* t = toknroll_tokenizer_load(custom_gguf());
    CHECK(t != NULL);
    int32_t ids[8];
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 8), 4);
    CHECK(same_ids(ids, HELLO, 4));
    toknroll_tokenizer_free(t);
}

/* configure's options are UTF-8: on Windows passed on in the ANSI code page, the one the runtime reads */
static void options_utf8(void) {
    const char* alias[] = {"-Dtoknroll.gguf.pre.caf\xC3\xA9=alias:gpt2"};
    CHECK_EQ(toknroll_configure(alias, 1), TOKNROLL_OK);
    const char* path = TMP("cafe.gguf");
    write_tiny_gguf(path, "caf\xC3\xA9");
    toknroll_tokenizer* t = toknroll_tokenizer_load(path);
    CHECK_STR(t ? "" : LE(NULL), "");
    toknroll_tokenizer_free(t);
}

/* OPTIONS_VAR set as UTF-8 text: a pre-tokenizer named "caf\xC3\xA9" (cafe, e acute), aliased only by it,
 * loads, the variable read as the runtime reads its options (on Windows, the ANSI code page) */
static void options_from_the_environment_utf8(void) {
    const char* path = TMP("cafe.gguf");
    write_tiny_gguf(path, "caf\xC3\xA9");
    toknroll_tokenizer* t = toknroll_tokenizer_load(path);
    CHECK_STR(t ? "" : LE(NULL), "");
    toknroll_tokenizer_free(t);
}

static void options_from_the_environment(void) {   /* OPTIONS_VAR alone */
    toknroll_tokenizer* t = toknroll_tokenizer_load(custom_gguf());
    CHECK(t != NULL);
    CHECK_EQ(toknroll_vocabulary_size(t), VOCAB_GGUF);
    toknroll_tokenizer_free(t);
}

static void options_environment_wins(void) {   /* OPTIONS_VAR aliases ctest-pre to nothing, after configure's */
    const char* alias[] = {ALIAS_CTEST};
    CHECK_EQ(toknroll_configure(alias, 1), TOKNROLL_OK);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(custom_gguf()), TOKNROLL_EINVAL, LE, "alias 'ctest-pre' points at unknown pre-tokenizer 'no-such-scheme'");
    CHECK_NULL_ERROR(toknroll_tokenizer_load(custom_gguf()), TOKNROLL_EINVAL, LE, "no-such-scheme");   /* each time */
    toknroll_tokenizer* t = toknroll_tokenizer_load(tiny_json());   /* a broken GGUF option spares the rest */
    CHECK(t != NULL);
    toknroll_tokenizer_free(t);
}

static void options_environment_rejected(void) {
    capture_stderr();
    CHECK_NULL_ERROR(toknroll_tokenizer_load(tiny_json()), TOKNROLL_ERROR, LE, "an option from " OPTIONS_VAR " was rejected");
    char* err = captured_stderr();
    CHECK_HAS(err, "NoSuchOptionAnywhere");
    free(err);
}

static void options_rejected_either_way(void) {   /* configure's are fine, the environment's not */
    const char* fine[] = {"-Dtoknroll.offline=true"};
    capture_stderr();
    CHECK_ERROR(toknroll_configure(fine, 1), TOKNROLL_ERROR, LE, "an option from configure or the environment was rejected");
}

/* ---- logging ---- */

typedef struct { int lines, replaced; int32_t level; char message[512]; void* user_data; } log_box;
static log_box seen;
static void keep_line(int32_t level, const char* message, void* user_data) {
    seen.lines++;
    seen.replaced += strstr(message, REPLACED) != NULL;
    seen.level = level;
    seen.user_data = user_data;
    snprintf(seen.message, sizeof seen.message, "%s", message);
}

/* a warning, then a GGUF to emit it, as the GGUF loader reads its options: what reaches the callback,
 * or stderr */
static void warn_once(void) {
    CHECK_EQ(toknroll_configure(REPLACE_GPT2, 1), TOKNROLL_OK);
    const char* path = TMP("warn.gguf");
    write_tiny_gguf(path, "llama-bpe");
    toknroll_tokenizer* t = toknroll_tokenizer_load(path);
    CHECK(t != NULL);
    toknroll_tokenizer_free(t);
}

/* the header's promise: beyond its log lines, which go to stderr, the library writes nothing to stdout */
static void writes_nothing_to_stdout(void) {
    capture_stdout();
    warn_once();   /* a warning, logged to stderr */
    toknroll_tokenizer* t = tiny();
    int32_t ids[8], offsets[2];
    const int64_t at[] = {0, 5};
    char out[16];
    toknroll_encode(t, "hello", 5, 0, ids, 8);
    toknroll_decode(t, ids, 4, 0, out, sizeof out, NULL);
    toknroll_encode_batch(t, "hello", at, 1, 0, ids, 8, offsets);
    toknroll_encode(t, "x", 1, 2, ids, 8);   /* refused */
    toknroll_tokenizer_free(t);
    toknroll_tokenizer_free(toknroll_tokenizer_load("./" TMP("missing.json")));
    char* written = captured_stdout();
    CHECK_STR(written, "");
    free(written);
}

static void log_set_once(void) {
    CHECK_EQ(toknroll_log_set(keep_line, &seen), TOKNROLL_OK);
    CHECK_CLEAR(LE);
    CHECK_ERROR(toknroll_log_set(keep_line, &seen), TOKNROLL_EINVAL, LE, "toknroll_log_set must come once, before any other call into the library");
    CHECK_ERROR(toknroll_log_set(NULL, NULL), TOKNROLL_EINVAL, LE, "must come once");
    warn_once();                                 /* log_set starts nothing: configure follows */
    CHECK_ERROR(toknroll_log_set(keep_line, &seen), TOKNROLL_EINVAL, LE, "toknroll_log_set must come once");
    CHECK_EQ(seen.lines, 1);                     /* the first set holds */
}

static void log_set_after_a_call(void) {
    toknroll_tokenizer_free(tiny());
    CHECK_ERROR(toknroll_log_set(keep_line, NULL), TOKNROLL_EINVAL, LE, "toknroll_log_set must come once");
}

static void log_set_after_configure(void) {
    CHECK_EQ(toknroll_configure(NULL, 0), TOKNROLL_OK);
    CHECK_ERROR(toknroll_log_set(keep_line, NULL), TOKNROLL_EINVAL, LE, "toknroll_log_set must come once");
}

static void log_to_stderr_by_default(void) {
    capture_stderr();
    warn_once();
    char* err = captured_stderr();
    CHECK_HAS(err, PROGRAM ": warn: com.qxotic.toknroll.gguf.GGUFTokenizerLoader: " REPLACED);
    const char* end = err ? strstr(err, NL) : NULL;
    CHECK(end && !end[strlen(NL)]);   /* one line, ended: its first line end is its last character */
    free(err);
}

static void log_to_the_callback(void) {
    static int tag;
    capture_stderr();
    CHECK_EQ(toknroll_log_set(keep_line, &tag), TOKNROLL_OK);
    warn_once();
    CHECK_EQ(seen.lines, 1);
    CHECK_EQ(seen.level, TOKNROLL_LOG_WARN);
    CHECK(seen.user_data == &tag);
    CHECK_HAS(seen.message, "com.qxotic.toknroll.gguf.GGUFTokenizerLoader: " REPLACED);
    CHECK(!strchr(seen.message, '\n'));          /* no newline */
    CHECK(strncmp(seen.message, PROGRAM ":", strlen(PROGRAM ":")));   /* nor stderr's prefix */
    char* err = captured_stderr();
    CHECK_STR(err, "");                          /* stderr untouched */
    free(err);
}

static void log_null_is_silence(void) {   /* run with LOG_VAR=debug: NULL wins */
    capture_stderr();
    CHECK_EQ(toknroll_log_set(NULL, NULL), TOKNROLL_OK);
    warn_once();
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void log_level_counts(int want) {
    capture_stderr();
    CHECK_EQ(toknroll_log_set(keep_line, NULL), TOKNROLL_OK);
    warn_once();
    CHECK_EQ(seen.replaced, want);   /* the line this test owns, whatever else a level lets through */
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void stderr_quiet(void) {   /* no callback: the variable filters stderr's lines the same way */
    capture_stderr();
    warn_once();
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

/* the callback may call last_error and free, even while the library starts: the unknown value's warning
 * comes from inside the start, and must not hang there (LOG_VAR=verbose) */
static toknroll_tokenizer* spare;
static int called_in;
static void call_in(int32_t level, const char* message, void* user_data) {
    (void) level; (void) message; (void) user_data;
    toknroll_tokenizer* t = spare;
    spare = NULL;   /* first: the free may log too */
    LE(NULL);
    called_in++;
    toknroll_tokenizer_free(t);
}

static void log_callback_calls_in(void) {
#ifndef _WIN32
    alarm(60);   /* a hang fails the scenario */
#endif
    CHECK_EQ(toknroll_log_set(call_in, NULL), TOKNROLL_OK);
    CHECK_EQ(toknroll_configure(REPLACE_GPT2, 1), TOKNROLL_OK);
    spare = tiny();                  /* the start's warning called in here */
    CHECK(called_in >= 1);
    CHECK(spare != NULL);            /* the first line came before spare was loaded */
    const char* path = TMP("calls-in.gguf");
    write_tiny_gguf(path, "gpt2");
    toknroll_tokenizer* t = toknroll_tokenizer_load(path);   /* the GGUF loader's warning frees spare, inside this call */
    CHECK(t != NULL);
    CHECK(spare == NULL && called_in >= 2);
    int32_t ids[8];
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 8), 4);   /* the outer call, and the library, carry on */
    toknroll_tokenizer_free(t);
}
static void log_filtered(void) { log_level_counts(0); }   /* off, error, " Error ": a warning filtered */
static void log_passed(void) { log_level_counts(1); }     /* warn, debug, and "" (the default, warn, no complaint) */

static void log_unknown_value(void) {   /* LOG_VAR=verbose: warn applies, and one warning says so */
    capture_stderr();
    CHECK_EQ(toknroll_log_set(keep_line, NULL), TOKNROLL_OK);
    toknroll_tokenizer_free(tiny());
    CHECK_EQ(seen.lines, 1);
    CHECK_EQ(seen.level, TOKNROLL_LOG_WARN);
    CHECK_STR(seen.message, PROGRAM ".native: " LOG_VAR "=verbose is not off, error, warn, info or debug: warn applies");
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void log_unknown_value_on_stderr(void) {
    capture_stderr();
    toknroll_tokenizer_free(tiny());
    char* err = captured_stderr();
    CHECK_STR(err, PROGRAM ": warn: " PROGRAM ".native: " LOG_VAR "=verbose is not off, error, warn, info or debug: warn applies" NL);
    free(err);
}

/* ---- load: Sources in toknroll.h, row by row ---- */

static void load_paths(void) {
    toknroll_tokenizer* t = toknroll_tokenizer_load(tiny_json());
    CHECK(t != NULL);
    CHECK_CLEAR(LE);
    toknroll_tokenizer* again = toknroll_tokenizer_load(tiny_json());   /* a second handle, as good */
    CHECK(again != NULL && again != t);
    toknroll_tokenizer_free(t);
    CHECK_EQ(toknroll_vocabulary_size(again), VOCAB_JSON);
    toknroll_tokenizer_free(again);

    CHECK_NULL_ERROR(toknroll_tokenizer_load(NULL), TOKNROLL_EINVAL, LE, "source is NULL");
    CHECK_NULL_ERROR(toknroll_tokenizer_load(""), TOKNROLL_EINVAL, LE, "source is empty");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("  \t"), TOKNROLL_EINVAL, LE, "source is empty");
    /* no shell reads a C string: ~ is a folder's name, never the home folder nor a hub owner */
    CHECK_NULL_ERROR(toknroll_tokenizer_load("~/ctest-none/tokenizer.json"), TOKNROLL_ENOTFOUND, LE, "~ is not expanded");
    CHECK_NULL_ERROR(toknroll_tokenizer_load(TMP("missing.json")), TOKNROLL_ENOTFOUND, LE, "does not exist");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("./ctest-missing.json"), TOKNROLL_ENOTFOUND, LE, "does not exist");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("~/ctest-missing/tokenizer.json"), TOKNROLL_ENOTFOUND, LE, "does not exist");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("owner/ctest-missing.gguf"), TOKNROLL_ENOTFOUND, LE, "does not exist");   /* a path, no repository */
    CHECK_NULL_ERROR(toknroll_tokenizer_load("ctest-mydir/tokenizer.json"), TOKNROLL_ENOTFOUND, LE, "ctest-mydir/tokenizer.json does not exist");   /* nor one */
    CHECK_NULL_ERROR(toknroll_tokenizer_load("ctest-missing.json"), TOKNROLL_EINVAL, LE, "is neither a file here nor a reference");
}

#ifndef _WIN32
/* a FIFO is refused, never opened: opening one with no writer blocks forever */
/* a path is UTF-8 on every platform: a name beyond ASCII loads, and a missing one is named back as given */
static void load_utf8_path(void) {
    const char* path = TMP("caf\xC3\xA9 \xE6\xB5\x8B.json");   /* "cafe" with e acute, a space, a CJK character */
    write_tiny_json(TMP("ascii.json"));
    rename_utf8(TMP("ascii.json"), path);
    toknroll_tokenizer* t = toknroll_tokenizer_load(path);
    CHECK(t != NULL);
    CHECK_EQ(toknroll_vocabulary_size(t), VOCAB_JSON);
    toknroll_tokenizer_free(t);
    CHECK_NULL_ERROR(toknroll_tokenizer_load("./" TMP("\xC3\xBC-missing.json")), TOKNROLL_ENOTFOUND, LE, "\xC3\xBC-missing.json does not exist");
}

static void load_fifo(void) {
    alarm(30);   /* the regression hangs: a hang fails the scenario */
    unlink(TMP("fifo"));
    CHECK(mkfifo(TMP("fifo"), 0600) == 0);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(TMP("fifo")), TOKNROLL_EINVAL, LE, "is not a regular file");
    unlink(TMP("fifo"));
}
#endif

static void load_folders(void) {
    const char* with = TMP("with");
    const char* without = TMP("without");
    make_dir(with);
    make_dir(without);
    write_tiny_json(TMP("with/tokenizer.json"));
    toknroll_tokenizer* t = toknroll_tokenizer_load(with);   /* a folder: its tokenizer.json */
    CHECK(t != NULL);
    CHECK_EQ(toknroll_vocabulary_size(t), VOCAB_JSON);
    toknroll_tokenizer_free(t);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(without), TOKNROLL_ENOTFOUND, LE, "holds no tokenizer.json");
}

static void load_by_content(void) {
    /* the content decides, whatever the name */
    const char* gguf_as_json = TMP("model.json");
    const char* json_as_gguf = TMP("tokenizer.gguf");
    const char* blob = TMP("sha256-3f1a");
    write_tiny_gguf(gguf_as_json, "gpt2");
    write_tiny_gguf(blob, "gpt2");
    write_tiny_json(json_as_gguf);
    toknroll_tokenizer* a = toknroll_tokenizer_load(gguf_as_json);
    toknroll_tokenizer* b = toknroll_tokenizer_load(json_as_gguf);
    toknroll_tokenizer* c = toknroll_tokenizer_load(blob);
    CHECK(a && b && c);
    CHECK_EQ(toknroll_vocabulary_size(a), VOCAB_GGUF);
    CHECK_EQ(toknroll_vocabulary_size(b), VOCAB_JSON);
    CHECK_EQ(toknroll_vocabulary_size(c), VOCAB_GGUF);
    toknroll_tokenizer_free(a);
    toknroll_tokenizer_free(b);
    toknroll_tokenizer_free(c);

    /* a file of no known kind */
    const char* garbage = TMP("garbage.bin");
    write_file(garbage, "\x01\x02 not a tokenizer", 18);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(garbage), TOKNROLL_EINVAL, LE, "is not a tokenizer");
    const char* empty = TMP("empty.json");
    write_file(empty, "", 0);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(empty), TOKNROLL_EINVAL, LE, "is not a tokenizer");
    const char* array = TMP("array.json");
    write_file(array, "[1, 2]", 6);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(array), TOKNROLL_EINVAL, LE, "is not a tokenizer");
    /* a known kind, malformed */
    const char* object = TMP("object.json");
    write_file(object, "{\"a\": 1}", 8);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(object), TOKNROLL_EINVAL, LE, "");
    const char* cut = TMP("cut.gguf");
    write_file(cut, "GGUF\x03\x00\x00\x00", 8);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(cut), TOKNROLL_EINVAL, LE, "the file is truncated");
    const char* json_cut = TMP("cut.json");
    write_file(json_cut, "{\"model\": {\"type\": \"BPE\"", 25);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(json_cut), TOKNROLL_EINVAL, LE, "Line 1, Column 25");
    const char* bom = TMP("bom.json");   /* refused as Hugging Face refuses it, by name */
    write_file(bom, "\xEF\xBB\xBF{}", 5);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(bom), TOKNROLL_EINVAL, LE, "starts with a UTF-8 byte order mark");
}

static void load_references(void) {
    /* offline, over an empty cache: a reference resolves or fails here, never on the network */
    char root[700];
    snprintf(root, sizeof root, "-Dtoknroll.cache.root=%s", TMP("cache"));
    const char* offline[] = {"-Dtoknroll.offline=true", root};
    CHECK_EQ(toknroll_configure(offline, 2), TOKNROLL_OK);
    CHECK_NULL_ERROR(toknroll_tokenizer_load("ctest-nobody/no-such-repo"), TOKNROLL_ENOTFOUND, LE, "");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("ctest-nobody/no-such-GGUF:Q8_0"), TOKNROLL_ENOTFOUND, LE, "");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("hf.co/ctest-nobody/no-such-repo"), TOKNROLL_ENOTFOUND, LE, "");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("huggingface.co/ctest-nobody/no-such-repo@v1"), TOKNROLL_ENOTFOUND, LE, "");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("modelscope.cn/ctest-nobody/no-such-repo"), TOKNROLL_ENOTFOUND, LE, "");
    /* the file named as the reference spells it, no Java null in it; a bare reference with a path spells a
     * relative path too, which would have won had it existed: the hint says how one is written */
    CHECK_NULL_ERROR(toknroll_tokenizer_load("ctest-nobody/repo/sub/x.gguf"), TOKNROLL_ENOTFOUND, LE, "ctest-nobody/repo/sub/x.gguf is not cached");
    CHECK_HAS(LE(NULL), "a local path is written './ctest-nobody/repo/sub/x.gguf'");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("ctest-mydir/sub/tokenizer.json"), TOKNROLL_ENOTFOUND, LE, "is not cached");
    CHECK_HAS(LE(NULL), "a local path is written './ctest-mydir/sub/tokenizer.json'");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("hf.co/ctest-nobody/repo/sub/x.gguf"), TOKNROLL_ENOTFOUND, LE, "is not cached");
    CHECK(!strstr(LE(NULL), "a local path"));   /* a host written: no path it could be */
    /* malformed: refused as spelled, before any lookup */
    CHECK_NULL_ERROR(toknroll_tokenizer_load("owner/repo@"), TOKNROLL_EINVAL, LE, "empty revision");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("example.com/owner/repo"), TOKNROLL_EINVAL, LE, "unknown host");
    CHECK_HAS(LE(NULL), "a local path is written './example.com/owner/repo'");
    /* which is: a missing file, no host */
    CHECK_NULL_ERROR(toknroll_tokenizer_load("./example.com/owner/repo"), TOKNROLL_ENOTFOUND, LE, "does not exist");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("owner/repo/x-Q8_0.gguf:Q8_0"), TOKNROLL_EINVAL, LE, "already names a file");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("owner/repo:Q8_0:Q4_0"), TOKNROLL_EINVAL, LE, "a :quant comes once, last");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("owner/repo:Q8_0/x.gguf"), TOKNROLL_EINVAL, LE, "repository 'repo:Q8_0' holds a ':'");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("hf.co/owner"), TOKNROLL_EINVAL, LE, "owner/repo");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("hf.co/owner/../repo"), TOKNROLL_EINVAL, LE, "");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("a b/c"), TOKNROLL_EINVAL, LE, "no hub name has");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("owner/repo#x"), TOKNROLL_EINVAL, LE, "no hub name has");
    CHECK_NULL_ERROR(toknroll_tokenizer_load("owner/repo?x=1"), TOKNROLL_EINVAL, LE, "no hub name has");
}

/* TOKNROLL_CACHE_ROOT moves the store as -Dtoknroll.cache.root does: a file under it loads offline */
static void load_cache_root_from_the_environment(void) {
    const char* dirs[] = {TMP("envcache"), TMP("envcache/hf.co"), TMP("envcache/hf.co/ctest-env"),
                          TMP("envcache/hf.co/ctest-env/repo")};
    for (int i = 0; i < COUNT(dirs); i++) make_dir(dirs[i]);
    write_tiny_gguf(TMP("envcache/hf.co/ctest-env/repo/tiny.gguf"), "gpt2");
    const char* offline[] = {"-Dtoknroll.offline=true"};
    CHECK_EQ(toknroll_configure(offline, 1), TOKNROLL_OK);
    toknroll_tokenizer* t = toknroll_tokenizer_load("ctest-env/repo/tiny.gguf");
    CHECK_STR(t ? "" : LE(NULL), "");
    toknroll_tokenizer_free(t);
}

/* a file the Hugging Face hub cache holds is read in place, whatever the cache root: HF_HUB_CACHE is a
 * hand-made hub cache, the root another, empty folder, and the load is offline */
static void load_from_the_hub_cache(void) {
    const char* dirs[] = {TMP("hfhub"), TMP("hfhub/models--ctest-hf--repo"), TMP("hfhub/models--ctest-hf--repo/refs"),
                          TMP("hfhub/models--ctest-hf--repo/snapshots"),
                          TMP("hfhub/models--ctest-hf--repo/snapshots/c0ffee")};
    for (int i = 0; i < COUNT(dirs); i++) make_dir(dirs[i]);
    write_file(TMP("hfhub/models--ctest-hf--repo/refs/main"), "c0ffee", 6);
    write_tiny_gguf(TMP("hfhub/models--ctest-hf--repo/snapshots/c0ffee/tiny.gguf"), "gpt2");
    char root[700];
    snprintf(root, sizeof root, "-Dtoknroll.cache.root=%s", TMP("emptyroot"));
    const char* options[] = {"-Dtoknroll.offline=true", root};
    CHECK_EQ(toknroll_configure(options, 2), TOKNROLL_OK);
    toknroll_tokenizer* t = toknroll_tokenizer_load("ctest-hf/repo/tiny.gguf");
    CHECK_STR(t ? "" : LE(NULL), "");
    toknroll_tokenizer_free(t);
}

/* a valid tokenizer of a kind toknroll does not implement: the caller may fall back to another library */
static void load_unsupported(void) {
    const char* wordpiece = TMP("wordpiece.json");
    const char* json = "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"h\":0},\"unk_token\":\"h\"}}";
    write_file(wordpiece, json, strlen(json));
    CHECK_NULL_ERROR(toknroll_tokenizer_load(wordpiece), TOKNROLL_EUNSUPPORTED, LE, "tokenizer.json:model.type: WordPiece");
}

/* ---- encode ---- */

static void encode(void) {
    toknroll_tokenizer* t = tiny();
    int32_t ids[64];
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 64), 4);
    CHECK(same_ids(ids, HELLO, 4));
    CHECK_CLEAR(LE);
    CHECK_EQ(toknroll_encode(t, "hello world", 5, 0, ids, 64), 4);   /* length bytes, not to the NUL */
    CHECK(same_ids(ids, HELLO, 4));
    /* capacity: 0 sizes, less gets a prefix and the whole count, nothing written past it */
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, NULL, 0), 4);
    for (int i = 0; i < 8; i++) ids[i] = -7;
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 2), 4);
    CHECK(ids[0] == ID_HE && ids[1] == 108 && ids[2] == -7);
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 4), 4);
    /* empty text, NULL with length 0 included */
    CHECK_EQ(toknroll_encode(t, "", 0, 0, ids, 64), 0);
    CHECK_EQ(toknroll_encode(t, NULL, 0, 0, ids, 64), 0);
    CHECK_EQ(toknroll_encode(t, NULL, 0, 0, NULL, 0), 0);
    /* any byte: a NUL inside, a character in two bytes, a byte that is no UTF-8 */
    CHECK_EQ(toknroll_encode(t, "a\0b", 3, 0, ids, 64), 3);
    CHECK(ids[0] == 'a' && ids[1] == 0 && ids[2] == 'b');
    CHECK_EQ(toknroll_encode(t, "\xC3\xA9", 2, 0, ids, 64), 2);
    CHECK(ids[0] == 0xC3 && ids[1] == 0xA9);
    /* errors */
    CHECK_ERROR(toknroll_encode(NULL, "x", 1, 0, ids, 64), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    CHECK_ERROR(toknroll_encode(t, NULL, 3, 0, ids, 64), TOKNROLL_EINVAL, LE, "text is NULL");
    CHECK_ERROR(toknroll_encode(t, "x", 1, 0, ids, -1), TOKNROLL_EINVAL, LE, "token_capacity must not be negative");
    CHECK_ERROR(toknroll_encode(t, "x", 1, 0, NULL, 4), TOKNROLL_EINVAL, LE, "out is NULL");
    CHECK_ERROR(toknroll_encode(t, "x", 1, 2, ids, 64), TOKNROLL_EINVAL, LE, "unknown flags 0x2");
    CHECK_ERROR(toknroll_encode(t, "x", 1, 0x80000000u, ids, 64), TOKNROLL_EINVAL, LE, "unknown flags 0x80000000");
    CHECK_ERROR(toknroll_encode(t, "x", (int64_t) 1 << 32, 0, ids, 64), TOKNROLL_EINVAL, LE, "over 2 GiB");
    CHECK_ERROR(toknroll_encode(t, "x", -1, 0, ids, 64), TOKNROLL_EINVAL, LE, "length must not be negative");
    CHECK_ERROR(toknroll_encode(t, "x", INT64_MIN, 0, NULL, 0), TOKNROLL_EINVAL, LE, "length must not be negative");
    /* just under 2^31: refused before any read, where the copy's slack once wrapped past the array */
    CHECK_ERROR(toknroll_encode(t, "x", INT32_MAX - 10, 0, ids, 64), TOKNROLL_EINVAL, LE, "text is over 2 GiB");
    toknroll_tokenizer_free(t);
}

static void encode_specials(void) {
    toknroll_tokenizer* t = tiny();
    int32_t ids[64];
    /* by default a spelled special token is text */
    CHECK_EQ(toknroll_encode(t, "a<|end|>b", 9, 0, ids, 64), 9);
    CHECK(ids[0] == 'a' && ids[1] == '<' && ids[8] == 'b');
    CHECK_EQ(toknroll_encode(t, "a<|end|>b", 9, TOKNROLL_SPECIALS, ids, 64), 3);
    CHECK(ids[0] == 'a' && ids[1] == ID_END && ids[2] == 'b');
    CHECK_EQ(toknroll_encode(t, "<|end|>", 7, TOKNROLL_SPECIALS, ids, 64), 1);
    CHECK_EQ(ids[0], ID_END);
    CHECK_EQ(toknroll_encode(t, "<|end|><|end|>", 14, TOKNROLL_SPECIALS, ids, 64), 2);
    /* an added token that is not special stays whole too: its type is user-defined, not normal */
    CHECK_EQ(toknroll_encode(t, "x<user>y", 8, TOKNROLL_SPECIALS, ids, 64), 3);
    CHECK(ids[0] == 'x' && ids[1] == ID_USER && ids[2] == 'y');
    CHECK_EQ(toknroll_encode(t, "x<user>y", 8, 0, ids, 64), 8);
    /* a special cut short is text */
    CHECK_EQ(toknroll_encode(t, "<|end|", 6, TOKNROLL_SPECIALS, ids, 64), 6);
    toknroll_tokenizer_free(t);
}

/* a failure reads as one line of at most 1 KB, in the library's words: an input echoed whole, a
 * Java class name or GraalVM's "Stack trace is imprecise" placeholder is a regression */
static void error_messages(void) {
    enum { BIG = 200000 };
    char* big = (char*) malloc(BIG);
    memset(big, '[', BIG);
    big[0] = '{';
    write_file(TMP("long.json"), big, BIG);   /* its parser echoes the line */
    free(big);
    CHECK(toknroll_tokenizer_load(TMP("long.json")) == NULL);
    CHECK(strlen(LE(NULL)) <= 1024 && !strchr(LE(NULL), '\n'));
    FILE* f = fopen(TMP("type.gguf"), "wb");   /* a metadata value of type -1 */
    fwrite("GGUF", 1, 4, f);
    gguf_u32(f, 3); gguf_u64(f, 0); gguf_u64(f, 1); gguf_str(f, "a"); gguf_u32(f, 0xFFFFFFFFu);
    fclose(f);
    CHECK_NULL_ERROR(toknroll_tokenizer_load(TMP("type.gguf")), TOKNROLL_EINVAL, LE, "Invalid metadata value type index: -1");
    CHECK(!strstr(LE(NULL), "Stack trace is imprecise"));
#ifndef _WIN32
    write_file(TMP("locked.json"), "{}", 2);
    chmod(TMP("locked.json"), 0);
    if (geteuid() != 0) {   /* root reads anything */
        CHECK_NULL_ERROR(toknroll_tokenizer_load(TMP("locked.json")), TOKNROLL_ERROR, LE, "locked.json: permission denied");
        CHECK(!strstr(LE(NULL), "java."));
    }
    chmod(TMP("locked.json"), 0600);
#endif
}

/* ---- decode ---- */

static void decode(void) {
    toknroll_tokenizer* t = tiny();
    char out[64];
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 0, out, 64, NULL), 5);
    CHECK_STR(out, "hello");
    CHECK_CLEAR(LE);
    /* capacity: 0 sizes; else the tokens that fit whole in capacity - 1 bytes and a NUL, never part
     * of a token, decoded_tokens saying how many, the whole size returned, below capacity exactly when the
     * text is whole */
    int32_t decoded_tokens = -7;
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 0, NULL, 0, NULL), 5);
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 0, NULL, 0, &decoded_tokens), 5);
    CHECK_EQ(decoded_tokens, 0);
    static const char* const cut[] = {"", "", "he", "hel", "hell", "hello"};   /* by capacity 1..6 */
    static const int32_t fit[] = {0, 0, 1, 2, 3, 4};
    for (int64_t capacity = 1; capacity <= 6; capacity++) {
        memset(out, 'Z', sizeof out);
        CHECK_EQ(toknroll_decode(t, HELLO, 4, 0, out, capacity, NULL), 5);
        CHECK_STR(out, cut[capacity - 1]);    /* at capacity 2, "he" does not fit in one byte */
        memset(out, 'Z', sizeof out);
        decoded_tokens = -7;
        CHECK_EQ(toknroll_decode(t, HELLO, 4, 0, out, capacity, &decoded_tokens), 5);
        CHECK_STR(out, cut[capacity - 1]);
        CHECK_EQ(decoded_tokens, fit[capacity - 1]);
    }
    memset(out, 'Z', sizeof out);
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 0, out, 6, NULL), 5);
    CHECK_EQ(out[6], 'Z');                    /* nothing past the NUL */
    /* no ids */
    CHECK_EQ(toknroll_decode(t, NULL, 0, 0, out, 64, NULL), 0);
    CHECK_STR(out, "");
    CHECK_EQ(toknroll_decode(t, NULL, 0, 0, NULL, 0, NULL), 0);
    decoded_tokens = -7;
    CHECK_EQ(toknroll_decode(t, NULL, 0, 0, out, 64, &decoded_tokens), 0);
    CHECK_EQ(decoded_tokens, 0);
    /* the tokens' own bytes: half a character, a NUL inside, a special token's spelling */
    int32_t half[] = {0xE2};
    CHECK_EQ(toknroll_decode(t, half, 1, 0, out, 64, NULL), 1);
    CHECK((unsigned char) out[0] == 0xE2 && out[1] == 0);
    int32_t nul[] = {'a', 0, 'b'};
    CHECK_EQ(toknroll_decode(t, nul, 3, 0, out, 64, NULL), 3);
    CHECK(!memcmp(out, "a\0b", 4));
    int32_t special[] = {ID_END, ID_USER};
    CHECK_EQ(toknroll_decode(t, special, 2, 0, out, 64, NULL), 13);
    CHECK_STR(out, "<|end|><user>");
    /* start: ids[start, count) in the whole; start == count is no ids; only those ids are checked */
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 1, out, 64, NULL), 3);
    CHECK_STR(out, "llo");
    decoded_tokens = -7;
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 1, out, 3, &decoded_tokens), 3);
    CHECK_STR(out, "ll");
    CHECK_EQ(decoded_tokens, 2);
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 4, out, 64, &decoded_tokens), 0);
    CHECK_STR(out, "");
    CHECK_EQ(decoded_tokens, 0);
    CHECK_EQ(toknroll_decode(t, NULL, 0, 0, NULL, 0, NULL), 0);
    int32_t bad_then_good[] = {-1, 'a'};
    CHECK_EQ(toknroll_decode(t, bad_then_good, 2, 1, out, 64, NULL), 1);   /* ids before start are not read */
    CHECK_STR(out, "a");
    /* errors: decoded_tokens is left as it was */
    decoded_tokens = -7;
    int32_t outside[] = {'a', VOCAB_JSON};
    CHECK_ERROR(toknroll_decode(t, outside, 2, 0, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "no token with id 259, at ids[1]");
    int32_t negative[] = {-1};
    CHECK_ERROR(toknroll_decode(t, negative, 1, 0, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "no token with id -1, at ids[0]");
    CHECK_ERROR(toknroll_decode(t, negative, 1, 0, NULL, 0, &decoded_tokens), TOKNROLL_EINVAL, LE, "no token with id -1");   /* sizing checks too */
    CHECK_ERROR(toknroll_decode(NULL, HELLO, 4, 0, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    CHECK_ERROR(toknroll_decode(t, HELLO, -1, 0, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "count must not be negative");
    CHECK_ERROR(toknroll_decode(t, NULL, 2, 0, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "ids is NULL");
    CHECK_ERROR(toknroll_decode(t, HELLO, 4, 0, out, -1, &decoded_tokens), TOKNROLL_EINVAL, LE, "byte_capacity must not be negative");
    CHECK_ERROR(toknroll_decode(t, HELLO, 4, 0, NULL, 8, &decoded_tokens), TOKNROLL_EINVAL, LE, "out is NULL");
    CHECK_ERROR(toknroll_decode(t, HELLO, 4, -1, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "start -1 is outside [0, count 4]");
    CHECK_ERROR(toknroll_decode(t, HELLO, 4, 5, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "start 5 is outside [0, count 4]");
    CHECK_ERROR(toknroll_decode(t, outside, 2, 1, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "no token with id 259, at ids[1]");   /* absolute */
    CHECK_ERROR(toknroll_decode(t, HELLO, INT32_MAX, 0, out, 64, &decoded_tokens), TOKNROLL_EINVAL, LE, "ids are over 2 GiB");   /* before any read */
    CHECK_EQ(decoded_tokens, -7);
    toknroll_tokenizer_free(t);
}

/* every start and capacity of a text's ids: out holds exactly the bytes ids[start, start + decoded)
 * have in the whole text, as many tokens as fit in capacity - 1, and they are all the rest exactly
 * when that rest is whole */
static int64_t prefix(toknroll_tokenizer* t, const int32_t* ids, int32_t k) {
    return toknroll_decode(t, ids, k, 0, NULL, 0, NULL);   /* the bytes of the text's first k ids */
}

static void check_cuts(toknroll_tokenizer* t, const int32_t* ids, int32_t n, const char* text) {
    int64_t size = (int64_t) strlen(text);
    char* out = (char*) malloc((size_t) size + 1);
    int wrong = 0;
    for (int32_t start = 0; start <= n; start++) {
        int64_t at = prefix(t, ids, start), rest = size - at;
        for (int64_t capacity = 1; capacity <= rest + 1; capacity++) {
            int32_t k = -7;
            int64_t got = toknroll_decode(t, ids, n, start, out, capacity, &k);
            int64_t kept = k < 0 || start + k > n ? -1 : prefix(t, ids, start + k) - at;
            int64_t more = k >= 0 && start + k < n ? prefix(t, ids, start + k + 1) - at : rest;
            wrong += got != rest || kept < 0 || kept >= capacity || (int64_t) strlen(out) != kept
                     || memcmp(out, text + at, (size_t) kept) || (start + k == n) != (rest < capacity)
                     || (start + k < n && more < capacity);   /* the next id would not have fit */
        }
    }
    CHECK_EQ(wrong, 0);
    free(out);
}

static void decode_cuts(void) {
    toknroll_tokenizer* t = tiny();
    static const char* texts[] = {"Hello, world!", "h\xC3\xA9llo <|end|> w\xC3\xB6rld", "hehehe", "\xF0\x9F\x8E\xB8 guitar"};
    for (int i = 0; i < COUNT(texts); i++) {
        int32_t ids[64];
        int32_t n = toknroll_encode(t, texts[i], strlen(texts[i]), TOKNROLL_SPECIALS, ids, 64);
        CHECK(n > 0);
        check_cuts(t, ids, n, texts[i]);
    }
    toknroll_tokenizer_free(t);
}

/* a text decoded through a small buffer, each call starting where the last stopped, and streamed
 * a token at a time: the pieces add up to the text, the spaces SentencePiece keeps in its tokens too */
static void check_pieces(toknroll_tokenizer* t, const int32_t* ids, int32_t n, const char* text) {
    char whole[256] = "", piece[9];
    for (int32_t start = 0, k = 0; start < n; start += k) {
        CHECK(toknroll_decode(t, ids, n, start, piece, sizeof piece, &k) >= 0);
        CHECK(k > 0);   /* no token of these texts is over 8 bytes */
        if (k <= 0) return;
        strcat(whole, piece);
    }
    CHECK_STR(whole, text);
    whole[0] = 0;
    for (int32_t i = 0; i < n; i++) {   /* streaming: the newest id, in the context of the reply so far */
        CHECK(toknroll_decode(t, ids, i + 1, i, piece, sizeof piece, NULL) >= 0);
        strcat(whole, piece);
    }
    CHECK_STR(whole, text);
}

static void decode_resumes(void) {
    toknroll_tokenizer* t = tiny();
    const char* text = "Tok'n'Roll \xF0\x9F\x8E\xB8 hello <|end|> hello again, hello";
    int32_t ids[128];
    int32_t n = toknroll_encode(t, text, strlen(text), TOKNROLL_SPECIALS, ids, 128);
    CHECK(n > 10);
    check_pieces(t, ids, n, text);
    toknroll_tokenizer_free(t);

    const char* path = TMP("metaspace.json");
    write_metaspace_json(path);
    t = toknroll_tokenizer_load(path);
    CHECK(t != NULL);
    if (!t) return;
    int32_t hi[8];
    CHECK_EQ(toknroll_encode(t, "hi there", 8, 0, hi, 8), 2);
    int32_t k = -7;
    char out[16];
    CHECK_EQ(toknroll_decode(t, hi, 2, 0, out, 4, &k), 8);
    CHECK_STR(out, "hi");                      /* " there" does not fit */
    CHECK_EQ(k, 1);
    CHECK_EQ(toknroll_decode(t, hi, 2, k, out, sizeof out, NULL), 6);
    CHECK_STR(out, " there");                  /* the rest of the text: its space kept */
    CHECK_EQ(toknroll_decode(t, hi + k, 2 - k, 0, out, sizeof out, NULL), 5);
    CHECK_STR(out, "there");                   /* a slice is a text of its own: its first space dropped */
    check_pieces(t, hi, 2, "hi there");
    check_cuts(t, hi, 2, "hi there");

    /* sizes past the library's 4096-id count chunks: a chunk counted on its own would drop its
       first token's space, a byte short per chunk */
    enum { WORDS = 3000 };
    char* many = (char*) malloc(WORDS * 9);
    char* end = many;
    for (int i = 0; i < WORDS; i++) end += sprintf(end, i ? " hi there" : "hi there");
    int64_t length = end - many;
    int32_t* words = (int32_t*) malloc(2 * WORDS * sizeof *words);
    CHECK_EQ(toknroll_encode(t, many, length, 0, words, 2 * WORDS), 2 * WORDS);
    CHECK_EQ(toknroll_decode(t, words, 2 * WORDS, 0, NULL, 0, NULL), length);
    const int32_t starts[] = {1, 4095, 4096, 4097, 2 * WORDS - 1};
    for (int i = 0; i < COUNT(starts); i++)   /* the ids before start and after it make the text */
        CHECK_EQ(toknroll_decode(t, words, starts[i], 0, NULL, 0, NULL)
                         + toknroll_decode(t, words, 2 * WORDS, starts[i], NULL, 0, NULL),
                 length);
    free(words);
    free(many);
    toknroll_tokenizer_free(t);
}

static void round_trips(void) {
    toknroll_tokenizer* t = tiny();
    static const char* texts[] = {
        "Hello, world!", "  leading and trailing  ", "line\nbreak\r\ntab\t", "h\xC3\xA9llo w\xC3\xB6rld",
        "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E", "\xF0\x9F\x8E\xB8 guitar", "", "<|end|> as text", "hehehe"};
    for (int i = 0; i < COUNT(texts); i++) {
        int32_t ids[128];
        char out[256];
        int32_t n = toknroll_encode(t, texts[i], strlen(texts[i]), 0, ids, 128);
        CHECK(n >= 0 && n <= 128);
        CHECK_EQ(toknroll_decode(t, ids, n, 0, out, sizeof out, NULL), (int64_t) strlen(texts[i]));
        CHECK_STR(out, texts[i]);
    }
    toknroll_tokenizer_free(t);
}

/* one token longer than the 64 KB a decode goes through at a time: the chunk grows to it, and over
 * capacity it is left out whole */
static void decode_a_token_past_the_chunk(void) {
    enum { LONG = 70000 };
    write_tiny_json(TMP("tiny-long.json"));
    FILE* f = fopen(TMP("tiny-long.json"), "rb");
    static char tiny[1 << 16], file[(1 << 16) + LONG + 64];
    size_t n = fread(tiny, 1, sizeof tiny - 1, f);
    fclose(f);
    const char* at = strstr(tiny, "\"added_tokens\":[") + strlen("\"added_tokens\":[");
    size_t k = (size_t) (at - tiny);
    memcpy(file, tiny, k);
    k += (size_t) sprintf(file + k, "{\"id\":300,\"special\":true,\"content\":\"");
    memset(file + k, 'x', LONG);
    k += LONG;
    k += (size_t) sprintf(file + k, "\"},");
    memcpy(file + k, at, n - (size_t) (at - tiny));
    write_file(TMP("long-token.json"), file, k + n - (size_t) (at - tiny));
    toknroll_tokenizer* t = toknroll_tokenizer_load(TMP("long-token.json"));
    CHECK(t != NULL);
    if (!t) return;
    const int32_t ids[] = {ID_HE, 300};   /* "he", then the long one */
    char* out = (char*) malloc(LONG + 3);
    int32_t decoded = -7;
    CHECK_EQ(toknroll_decode(t, ids, 2, 0, out, LONG + 3, &decoded), LONG + 2);
    CHECK_EQ(decoded, 2);
    CHECK(!memcmp(out, "he", 2) && out[2] == 'x' && out[LONG + 1] == 'x' && out[LONG + 2] == 0);
    CHECK_EQ(toknroll_decode(t, ids, 2, 0, out, LONG, &decoded), LONG + 2);   /* it does not fit: left out */
    CHECK_EQ(decoded, 1);
    CHECK_STR(out, "he");
    CHECK_EQ(toknroll_decode(t, ids, 2, 0, NULL, 0, NULL), LONG + 2);
    free(out);
    toknroll_tokenizer_free(t);
}

static void long_text(void) {   /* past the library's own buffers, whole and cut */
    toknroll_tokenizer* t = tiny();
    enum { REPEAT = 40000 };
    size_t size = REPEAT * 6;
    char* text = (char*) malloc(size + 1);
    for (int i = 0; i < REPEAT; i++) memcpy(text + i * 6, "hello ", 6);
    text[size] = 0;
    int32_t n = toknroll_encode(t, text, size, 0, NULL, 0);
    CHECK_EQ(n, REPEAT * 5);   /* he l l o Ġ */
    int32_t* ids = (int32_t*) malloc((size_t) n * sizeof *ids);
    CHECK_EQ(toknroll_encode(t, text, size, 0, ids, n), n);
    char* out = (char*) malloc(size + 1);
    CHECK_EQ(toknroll_decode(t, ids, n, 0, out, (int64_t) size + 1, NULL), (int64_t) size);
    CHECK(!memcmp(out, text, size + 1));
    int32_t decoded_tokens = -7;
    CHECK_EQ(toknroll_decode(t, ids, n, 0, out, (int64_t) size, &decoded_tokens), (int64_t) size);   /* one short: the last token out */
    CHECK_EQ((int64_t) strlen(out), (int64_t) size - 1);
    CHECK(!memcmp(out, text, size - 1));
    CHECK_EQ(decoded_tokens, n - 1);
    /* 70000 bytes, in the library's second 64 KB chunk: 11666 words of 5 tokens, then "hell" */
    CHECK_EQ(toknroll_decode(t, ids, n, 0, out, 70001, &decoded_tokens), (int64_t) size);
    CHECK_EQ((int64_t) strlen(out), 70000);
    CHECK(!memcmp(out, text, 70000));
    CHECK_EQ(decoded_tokens, 11666 * 5 + 3);
    free(out);
    free(ids);
    free(text);
    toknroll_tokenizer_free(t);
}

/* ---- encode_batch ---- */

#ifdef __linux__
#include <dirent.h>
/* the batch workers, by name (WORKER, the end of it Linux keeps): the runtime's own threads uncounted */
static int batch_workers(void) {
    int n = 0;
    DIR* d = opendir("/proc/self/task");
    for (struct dirent* e; d && (e = readdir(d));) {
        char path[300], name[32] = "";
        snprintf(path, sizeof path, "/proc/self/task/%s/comm", e->d_name);
        FILE* f = e->d_name[0] == '.' ? NULL : fopen(path, "r");
        if (f) { n += fgets(name, sizeof name, f) && strstr(name, WORKER); fclose(f); }
    }
    if (d) closedir(d);
    return n;
}

/* OPTIONS_VAR caps the cores at 2: only a batch with work to share starts the worker beside the caller. */
static void batch_threads_capped(void) {
    toknroll_tokenizer* t = tiny();
    CHECK_EQ(batch_workers(), 0);
    const char texts[] = "aaaabbbbccccddddeeeeffffgggghhhh";
    int64_t at[9];
    for (int i = 0; i <= 8; i++) at[i] = 4 * i;
    int32_t out[64], offsets[9];
    CHECK_EQ(toknroll_encode_batch(t, NULL, NULL, 0, 0, NULL, 0, NULL), 0);
    CHECK_EQ(batch_workers(), 0);
    int32_t single[4];
    CHECK_EQ(toknroll_encode(t, texts, at[1], 0, single, 4), 4);
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 1, 0, out, 64, offsets), 4);
    CHECK(same_ids(out, single, 4) && offsets[0] == 0 && offsets[1] == 4);
    CHECK_EQ(batch_workers(), 0);   /* a single task has nothing to share */
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 8, 0, out, 64, offsets), 32);
    for (int i = 0; i < 32; i++) CHECK_EQ(out[i], texts[i]);
    for (int i = 0; i <= 8; i++) CHECK_EQ(offsets[i], 4 * i);
    CHECK_EQ(batch_workers(), 1);   /* ActiveProcessorCount=2: one beside the caller */
    toknroll_tokenizer_free(t);
}
#endif

static void batch(void) {
    toknroll_tokenizer* t = tiny();
    const char texts[] = "hello" "" "a<|end|>b" "\xC3\xA9";
    const int64_t at[] = {0, 5, 5, 14, 16};
    int32_t out[64], offsets[5];
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 4, 0, out, 64, offsets), 15);
    CHECK(offsets[0] == 0 && offsets[1] == 4 && offsets[2] == 4 && offsets[3] == 13 && offsets[4] == 15);
    CHECK(same_ids(out, HELLO, 4) && out[4] == 'a' && out[5] == '<' && out[13] == 0xC3);
    CHECK_CLEAR(LE);
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 4, TOKNROLL_SPECIALS, out, 64, offsets), 9);
    CHECK(offsets[3] == 7 && out[5] == ID_END);
    /* capacity: 0 counts, less gets a prefix, offsets always whole */
    memset(offsets, 0, sizeof offsets);
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 4, 0, NULL, 0, offsets), 15);
    CHECK(offsets[1] == 4 && offsets[4] == 15);
    for (int i = 0; i < 16; i++) out[i] = -7;
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 4, 0, out, 6, offsets), 15);
    CHECK(same_ids(out, HELLO, 4) && out[4] == 'a' && out[5] == '<' && out[6] == -7);
    CHECK(offsets[4] == 15);
    /* offsets from any base: text i is [at[i], at[i + 1]) of texts */
    const int64_t later[] = {5, 10};
    CHECK_EQ(toknroll_encode_batch(t, "XXXXXhello", later, 1, 0, out, 64, offsets), 4);
    CHECK(same_ids(out, HELLO, 4) && offsets[1] == 4);
    /* nothing to encode */
    const int64_t none[] = {0};
    offsets[0] = 99;
    CHECK_EQ(toknroll_encode_batch(t, NULL, none, 0, 0, NULL, 0, offsets), 0);
    CHECK_EQ(offsets[0], 0);
    /* no text, so the arrays may be NULL, as a pointer with length 0 may */
    CHECK_EQ(toknroll_encode_batch(t, NULL, NULL, 0, 0, NULL, 0, NULL), 0);
    offsets[0] = 99;
    CHECK_EQ(toknroll_encode_batch(t, NULL, NULL, 0, 0, NULL, 0, offsets), 0);
    CHECK_EQ(offsets[0], 0);
    const int64_t empties[] = {0, 0, 0};
    CHECK_EQ(toknroll_encode_batch(t, NULL, empties, 2, 0, out, 64, offsets), 0);
    /* errors */
    CHECK_ERROR(toknroll_encode_batch(NULL, texts, at, 4, 0, out, 64, offsets), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    CHECK_ERROR(toknroll_encode_batch(t, texts, at, -1, 0, out, 64, offsets), TOKNROLL_EINVAL, LE, "count must not be negative");
    CHECK_ERROR(toknroll_encode_batch(t, texts, at, 4, 0, out, -1, offsets), TOKNROLL_EINVAL, LE, "token_capacity must not be negative");
    CHECK_ERROR(toknroll_encode_batch(t, texts, NULL, 4, 0, out, 64, offsets), TOKNROLL_EINVAL, LE, "text_offsets and offsets must not be NULL");
    CHECK_ERROR(toknroll_encode_batch(t, texts, at, 4, 0, out, 64, NULL), TOKNROLL_EINVAL, LE, "text_offsets and offsets must not be NULL");
    CHECK_ERROR(toknroll_encode_batch(t, texts, at, 4, 0, NULL, 64, offsets), TOKNROLL_EINVAL, LE, "out is NULL");
    CHECK_ERROR(toknroll_encode_batch(t, texts, at, 4, 4, out, 64, offsets), TOKNROLL_EINVAL, LE, "unknown flags 0x4");
    const int64_t back[] = {0, 5, 3};
    CHECK_ERROR(toknroll_encode_batch(t, texts, back, 2, 0, out, 64, offsets), TOKNROLL_EINVAL, LE, "text_offsets must not decrease, at 2");
    const int64_t below[] = {-1, 3};
    CHECK_ERROR(toknroll_encode_batch(t, texts, below, 1, 0, out, 64, offsets), TOKNROLL_EINVAL, LE, "text_offsets must not be negative");
    /* wild offsets are refused before any read: a difference that would wrap, a text past 2^31 bytes */
    const int64_t wrap[] = {INT64_MAX, INT64_MIN};
    CHECK_ERROR(toknroll_encode_batch(t, "x", wrap, 1, 0, out, 8, offsets), TOKNROLL_EINVAL, LE, "text_offsets must not decrease, at 1");
    const int64_t huge[] = {0, INT64_MAX};
    CHECK_ERROR(toknroll_encode_batch(t, "x", huge, 1, 0, out, 8, offsets), TOKNROLL_EINVAL, LE, "text 0 is over 2^31 bytes");
    const int64_t top[] = {INT64_MAX - 1, INT64_MAX};   /* consistent, but no address: once a crash */
    CHECK_ERROR(toknroll_encode_batch(t, "x", top, 1, 0, out, 8, offsets), TOKNROLL_EINVAL, LE, "past the end of memory");
    const int64_t under[] = {0, INT32_MAX - 10};   /* the copy's slack once wrapped past the array */
    CHECK_ERROR(toknroll_encode_batch(t, "x", under, 1, 0, out, 8, offsets), TOKNROLL_EINVAL, LE, "text 0 is over 2^31 bytes");
    /* refused before the count + 1 offsets are read */
    CHECK_ERROR(toknroll_encode_batch(t, "x", huge, INT32_MAX, 0, out, 8, offsets), TOKNROLL_EINVAL, LE, "count is over 2^31 texts");
    CHECK_ERROR(toknroll_encode_batch(t, NULL, at, 4, 0, out, 64, offsets), TOKNROLL_EINVAL, LE, "texts is NULL");
    toknroll_tokenizer_free(t);
}

static void batch_matches_encode(void) {   /* many texts, every core: each text's ids as encode makes them */
    toknroll_tokenizer* t = tiny();
    enum { TEXTS = 3000 };
    char* texts = (char*) malloc(TEXTS * 64);
    int64_t* at = (int64_t*) malloc((TEXTS + 1) * sizeof *at);
    at[0] = 0;
    for (int i = 0; i < TEXTS; i++) {
        int n = snprintf(texts + at[i], 64, "%s %d <|end|>%.*s", i % 3 ? "hello" : "h\xC3\xA9", i, i % 17, "abcdefghijklmnopq");
        at[i + 1] = at[i] + n;
    }
    int32_t* offsets = (int32_t*) malloc((TEXTS + 1) * sizeof *offsets);
    int32_t total = toknroll_encode_batch(t, texts, at, TEXTS, TOKNROLL_SPECIALS, NULL, 0, offsets);
    CHECK(total > 0);
    int32_t* out = (int32_t*) malloc((size_t) total * sizeof *out);
    CHECK_EQ(toknroll_encode_batch(t, texts, at, TEXTS, TOKNROLL_SPECIALS, out, total, offsets), total);
    int wrong = 0;
    for (int i = 0; i < TEXTS; i++) {
        int32_t ids[128];
        int32_t n = toknroll_encode(t, texts + at[i], (size_t) (at[i + 1] - at[i]), TOKNROLL_SPECIALS, ids, 128);
        if (n != offsets[i + 1] - offsets[i] || !same_ids(ids, out + offsets[i], n)) wrong++;
    }
    CHECK_EQ(wrong, 0);
    free(offsets);
    free(out);
    free(at);
    free(texts);
    toknroll_tokenizer_free(t);
}

/* ---- the vocabulary ---- */

static void vocabulary(void) {
    toknroll_tokenizer* t = tiny();
    CHECK_EQ(toknroll_vocabulary_size(t), VOCAB_JSON);
    CHECK_ERROR(toknroll_vocabulary_size(NULL), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    char out[16];
    /* token: the spelling as the vocabulary writes it, sized and cut as text */
    CHECK_EQ(toknroll_token(t, ID_HE, out, 16), 2);
    CHECK_STR(out, "he");
    CHECK_EQ(toknroll_token(t, ID_HE, NULL, 0), 2);
    CHECK_EQ(toknroll_token(t, ID_HE, out, 2), 2);
    CHECK_STR(out, "h");
    CHECK_EQ(toknroll_token(t, ID_HE, out, 1), 2);
    CHECK_STR(out, "");
    CHECK_EQ(toknroll_token(t, ' ', out, 16), 2);
    CHECK_STR(out, "\xC4\xA0");             /* byte-level's space, "Ġ" */
    CHECK_EQ(toknroll_token(t, ID_END, out, 16), 7);
    CHECK_STR(out, "<|end|>");
    CHECK_ERROR(toknroll_token(t, VOCAB_JSON, out, 16), TOKNROLL_ENOTFOUND, LE, "no token with id 259");
    CHECK_ERROR(toknroll_token(t, -1, out, 16), TOKNROLL_ENOTFOUND, LE, "no token with id -1");
    CHECK_ERROR(toknroll_token(NULL, 0, out, 16), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    CHECK_ERROR(toknroll_token(t, 0, out, -1), TOKNROLL_EINVAL, LE, "byte_capacity must not be negative");
    CHECK_ERROR(toknroll_token(t, 0, NULL, 4), TOKNROLL_EINVAL, LE, "out is NULL");
    /* id: length bytes of a spelling */
    CHECK_EQ(toknroll_id(t, "he", 2), ID_HE);
    CHECK_EQ(toknroll_id(t, "hex", 2), ID_HE);
    CHECK_EQ(toknroll_id(t, "h", 1), 'h');
    CHECK_EQ(toknroll_id(t, "\xC4\xA0", 2), ' ');
    CHECK_EQ(toknroll_id(t, "<|end|>", 7), ID_END);
    CHECK_EQ(toknroll_id(t, "<user>", 6), ID_USER);
    CHECK_CLEAR(LE);
    CHECK_ERROR(toknroll_id(t, "nope", 4), TOKNROLL_ENOTFOUND, LE, "no token 'nope'");
    CHECK_ERROR(toknroll_id(t, " ", 1), TOKNROLL_ENOTFOUND, LE, "no token");   /* the text, not the spelling */
    CHECK_ERROR(toknroll_id(t, NULL, 0), TOKNROLL_ENOTFOUND, LE, "no token");
    CHECK_ERROR(toknroll_id(t, NULL, 1), TOKNROLL_EINVAL, LE, "token is NULL");
    CHECK_ERROR(toknroll_id(NULL, "he", 2), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    CHECK_ERROR(toknroll_id(t, "he", (int64_t) 1 << 32), TOKNROLL_EINVAL, LE, "over 2 GiB");
    CHECK_ERROR(toknroll_id(t, "he", -1), TOKNROLL_EINVAL, LE, "length must not be negative");
    toknroll_tokenizer_free(t);
}

static int one_type(int64_t flags) {
    static const int64_t types[] = {TOKNROLL_TOKEN_NORMAL, TOKNROLL_TOKEN_UNKNOWN, TOKNROLL_TOKEN_CONTROL,
                                    TOKNROLL_TOKEN_USER_DEFINED, TOKNROLL_TOKEN_UNUSED, TOKNROLL_TOKEN_BYTE};
    int set = 0;
    for (int i = 0; i < COUNT(types); i++) set += (flags & types[i]) != 0;
    return set == 1;
}

static void token_flags(void) {
    toknroll_tokenizer* t = tiny();
    CHECK_EQ(toknroll_token_flags(t, 'a'), TOKNROLL_TOKEN_NORMAL);
    CHECK_EQ(toknroll_token_flags(t, ID_HE), TOKNROLL_TOKEN_NORMAL);
    CHECK_EQ(toknroll_token_flags(t, ID_END), TOKNROLL_TOKEN_CONTROL);
    CHECK_EQ(toknroll_token_flags(t, ID_USER), TOKNROLL_TOKEN_USER_DEFINED);   /* every id: vocabulary_round_trip */
    CHECK_ERROR(toknroll_token_flags(t, VOCAB_JSON), TOKNROLL_ENOTFOUND, LE, "no token with id 259");
    CHECK_ERROR(toknroll_token_flags(t, -1), TOKNROLL_ENOTFOUND, LE, "no token with id -1");
    CHECK_ERROR(toknroll_token_flags(t, INT32_MIN), TOKNROLL_ENOTFOUND, LE, "no token");
    CHECK_ERROR(toknroll_token_flags(NULL, 0), TOKNROLL_EINVAL, LE, "tokenizer is NULL");
    toknroll_tokenizer_free(t);
}

static void gguf(void) {   /* the GGUF's own token types: all six */
    const char* path = TMP("tiny.gguf");
    write_tiny_gguf(path, "gpt2");
    toknroll_tokenizer* t = toknroll_tokenizer_load(path);
    CHECK(t != NULL);
    CHECK_EQ(toknroll_vocabulary_size(t), VOCAB_GGUF);
    CHECK_EQ(toknroll_token_flags(t, 'a'), TOKNROLL_TOKEN_NORMAL);
    CHECK_EQ(toknroll_token_flags(t, ID_UNK), TOKNROLL_TOKEN_UNKNOWN);
    CHECK_EQ(toknroll_token_flags(t, ID_END), TOKNROLL_TOKEN_CONTROL);
    CHECK_EQ(toknroll_token_flags(t, ID_USER), TOKNROLL_TOKEN_USER_DEFINED);
    CHECK_EQ(toknroll_token_flags(t, ID_UNUSED), TOKNROLL_TOKEN_UNUSED);
    CHECK_EQ(toknroll_token_flags(t, ID_BYTE), TOKNROLL_TOKEN_BYTE);
    int32_t ids[16];
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 16), 4);
    CHECK(same_ids(ids, HELLO, 4));
    CHECK_EQ(toknroll_encode(t, "a<|end|>b", 9, TOKNROLL_SPECIALS, ids, 16), 3);
    CHECK_EQ(ids[1], ID_END);
    char out[16];
    CHECK_EQ(toknroll_token(t, ID_UNUSED, out, 16), 8);
    CHECK_STR(out, "<unused>");
    toknroll_tokenizer_free(t);
}

/* ---- threads ---- */

typedef struct { toknroll_tokenizer* t; int index, wrong; } worker;
static void encode_worker(void* p) {
    worker* w = (worker*) p;
    for (int i = 0; i < 400; i++) {
        char text[64];
        int n = snprintf(text, sizeof text, "hello %d from %d <|end|>", i, w->index);
        int32_t ids[64];
        char back[128];
        int32_t count = toknroll_encode(w->t, text, (size_t) n, TOKNROLL_SPECIALS, ids, 64);
        if (count <= 0 || ids[0] != ID_HE || ids[count - 1] != ID_END) w->wrong++;
        if (toknroll_decode(w->t, ids, count, 0, back, sizeof back, NULL) != n || strcmp(back, text)) w->wrong++;
        int32_t code;
        LE(&code);
        if (code != TOKNROLL_OK) w->wrong++;
    }
}
static void load_worker(void* p) {
    worker* w = (worker*) p;
    for (int i = 0; i < 20; i++) {
        toknroll_tokenizer* own = toknroll_tokenizer_load(TMP("tiny.json"));
        if (!own || toknroll_vocabulary_size(own) != VOCAB_JSON) w->wrong++;
        toknroll_tokenizer_free(own);
    }
}
static void error_worker(void* p) {   /* failing calls on other threads leave everyone else's alone */
    worker* w = (worker*) p;
    for (int i = 0; i < 400; i++) {
        int32_t code;
        if (toknroll_token(w->t, -1, NULL, 0) != TOKNROLL_ENOTFOUND) w->wrong++;
        if (!strstr(LE(&code), "no token with id -1") || code != TOKNROLL_ENOTFOUND) w->wrong++;
    }
}

static void threads(void) {
    toknroll_tokenizer* t = tiny();
    enum { N = 10 };
    worker workers[N];
    thread_t running[N];
    for (int i = 0; i < N; i++) {
        workers[i] = (worker) {t, i, 0};
        running[i] = thread_run(i < 6 ? encode_worker : i < 8 ? load_worker : error_worker, &workers[i]);
    }
    int wrong = 0;
    for (int i = 0; i < N; i++) {
        thread_join(running[i]);
        wrong += workers[i].wrong;
    }
    CHECK_EQ(wrong, 0);
    toknroll_tokenizer_free(t);
}

/* ---- fork ---- */

#ifndef _WIN32
static int child_status(pid_t pid) {
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

static volatile int busy_stop;
static void busy_worker(void* p) {
    worker* w = (worker*) p;
    while (!busy_stop) {
        int32_t ids[16];
        if (toknroll_encode(w->t, "hello", 5, 0, ids, 16) != 4) w->wrong++;
    }
}

#ifdef TEST_LIBJINFER
/* libjinfer refuses a forked child: its threads stayed in the parent */
static int in_child(toknroll_tokenizer* t) {
    int32_t ids[16];
    int bad = 0;
    bad += toknroll_encode(t, "hello", 5, 0, ids, 16) != TOKNROLL_ERROR;
    bad += !strstr(LE(NULL), "fork");
    bad += toknroll_tokenizer_load(TMP("tiny.json")) != NULL;
    toknroll_tokenizer_free(t);   /* does nothing */
    bad += !*toknroll_version();
    return bad;
}
#else
/* libtoknroll's child keeps every tokenizer, and may load more; its batches run on the calling thread */
static int in_child(toknroll_tokenizer* t) {
    int32_t ids[16], offsets[3];
    int bad = 0;
    bad += toknroll_encode(t, "hello", 5, 0, ids, 16) != 4 || !same_ids(ids, HELLO, 4);
    const int64_t at[] = {0, 5, 10};
    bad += toknroll_encode_batch(t, "hellohello", at, 2, 0, ids, 16, offsets) != 8;
    toknroll_tokenizer* own = toknroll_tokenizer_load(TMP("tiny.json"));
    bad += own == NULL;
    toknroll_tokenizer_free(own);
    toknroll_tokenizer_free(t);
    return bad;
}
#endif

static void fork_children(void) {
    toknroll_tokenizer* t = tiny();
    alarm(120);   /* a hang fails the scenario */
    pid_t pid = fork();
    if (pid == 0) _exit(in_child(t));
    CHECK_EQ(child_status(pid), 0);
    /* forks while another thread is inside the library, again and again */
    worker w = {t, 0, 0};
    busy_stop = 0;
    thread_t busy = thread_run(busy_worker, &w);
    int failed = 0;
    for (int i = 0; i < 20; i++) {
        pid = fork();
        if (pid == 0) _exit(in_child(t));
        failed += child_status(pid) != 0;
    }
    busy_stop = 1;
    thread_join(busy);
    CHECK_EQ(failed, 0);
    CHECK_EQ(w.wrong, 0);
    int32_t ids[16];
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, 16), 4);   /* the parent carries on */
    toknroll_tokenizer_free(t);
}

/* forks while another thread runs the process's first batch, the one that starts the workers: fork once
 * held the lock a new worker lists itself under while it waited for that batch, and neither moved */
typedef struct { toknroll_tokenizer* t; volatile int total; } first;
static void first_batch(void* p) {
    first* f = (first*) p;
    int32_t ids[16], offsets[3];
    const int64_t at[] = {0, 5, 10};
    f->total = toknroll_encode_batch(f->t, "hellohello", at, 2, 0, ids, 16, offsets) + 1;   /* 0 while running */
}

static void fork_during_first_batch(void) {
    first f = {tiny(), 0};
    alarm(120);   /* the deadlock was a hang */
    thread_t batch = thread_run(first_batch, &f);
    while (!f.total) {
        pid_t pid = fork();
        if (pid == 0) _exit(0);
        CHECK_EQ(child_status(pid), 0);
    }
    thread_join(batch);
    CHECK_EQ(f.total - 1, 8);
    toknroll_tokenizer_free(f.t);
}
#endif

/* ---- the everyday paths, on the real tokenizers the cache holds (skipped, and said so, without) ---- */

#define QWEN "Qwen/Qwen3-0.6B"

#include "progress.h"   /* a download's progress: on unless the host says off, plain lines on a pipe, the board on a pty */
#define QWEN_VOCAB 151669
#define LFM_GGUF "LiquidAI/LFM2.5-350M-GGUF:Q8_0"

/* offline, so a hub reference resolves from the cache or not at all */
static void offline(void) {
    const char* options[] = {"-Dtoknroll.offline=true"};
    if (toknroll_configure(options, 1) != TOKNROLL_OK) { printf("    configure: %s\n", LE(NULL)); exit(1); }
}

/* a real tokenizer from the cache: skipped, and counted so, when it is not there */
static toknroll_tokenizer* cached(const char* ref) {
    toknroll_tokenizer* t = toknroll_tokenizer_load(ref);
    if (!t) skip("%s is not cached (%s)", ref, LE(NULL));
    return t;
}

#ifdef __linux__
#include <sys/resource.h>
/* under a virtual memory limit below the address space the runtime reserves for its heap, the start fails in
 * words: what it reserved, and the two ways out (the limit, or -XX:MaxHeapSize); the limit is this
 * process's own, set before the first call (macOS does not enforce RLIMIT_AS; Windows has none) */
static void address_space_too_small(void) {
    struct rlimit limit = {8ULL << 30, 8ULL << 30};
    if (setrlimit(RLIMIT_AS, &limit) != 0) skip("setrlimit: %s", strerror(errno));
    CHECK_NULL_ERROR(toknroll_tokenizer_load(tiny_json()), TOKNROLL_ERROR, LE, "could not reserve the address space");
    CHECK_HAS(LE(NULL), "-XX:MaxHeapSize");
}
#endif

/* a heap too small for a 151k-token vocabulary: out of memory names the knob, and the process goes on */
static void heap_too_small(void) {
    const char* options[] = {"-Dtoknroll.offline=true", "-XX:MaxHeapSize=64m"};   /* configure is once: both at once */
    CHECK_EQ(toknroll_configure(options, 2), TOKNROLL_OK);
    toknroll_tokenizer* t = toknroll_tokenizer_load(QWEN);
    if (t) { toknroll_tokenizer_free(t); FAIL_AT("%s loaded in 64 MB: raise the test's vocabulary", QWEN); return; }
    int32_t code = 0;
    const char* message = LE(&code);
    if (code == TOKNROLL_ENOTFOUND) skip("%s is not cached (%s)", QWEN, message);
    CHECK_EQ(code, TOKNROLL_ERROR);
    CHECK_HAS(message, "out of memory");
    CHECK_HAS(message, "-XX:MaxHeapSize");
    toknroll_tokenizer* tiny = toknroll_tokenizer_load(tiny_json());   /* the process is still usable; `small` is a Windows macro */
    CHECK(tiny != NULL);
    toknroll_tokenizer_free(tiny);
}

static void hub_reference(void) {
    offline();
    toknroll_tokenizer* t = cached(QWEN);
    CHECK_EQ(toknroll_vocabulary_size(t), QWEN_VOCAB);
    static const int32_t hello[] = {9707, 11, 1879, 0};
    int32_t ids[64];
    CHECK_EQ(toknroll_encode(t, "Hello, world!", 13, 0, ids, 64), 4);
    CHECK(same_ids(ids, hello, 4));
    char out[64];
    CHECK_EQ(toknroll_decode(t, hello, 4, 0, out, 64, NULL), 13);
    CHECK_STR(out, "Hello, world!");
    /* the same repository, spelled every way the grammar allows */
    static const char* spellings[] = {"hf.co/" QWEN, "huggingface.co/" QWEN, QWEN "/tokenizer.json", "  " QWEN "  ", QWEN "/"};
    for (int i = 0; i < COUNT(spellings); i++) {
        toknroll_tokenizer* same = toknroll_tokenizer_load(spellings[i]);
        CHECK(same != NULL);
        if (!same) { printf("    %s: %s\n", spellings[i], LE(NULL)); continue; }
        CHECK_EQ(toknroll_vocabulary_size(same), QWEN_VOCAB);
        toknroll_tokenizer_free(same);
    }
    /* a special token: its id, its type, whole with SPECIALS, text without */
    CHECK_EQ(toknroll_id(t, "<|im_end|>", 10), 151645);
    CHECK_EQ(toknroll_token_flags(t, 151645), TOKNROLL_TOKEN_CONTROL);
    CHECK_EQ(toknroll_encode(t, "<|im_end|>", 10, TOKNROLL_SPECIALS, ids, 64), 1);
    CHECK_EQ(ids[0], 151645);
    CHECK_EQ(toknroll_encode(t, "<|im_end|>", 10, 0, ids, 64), 6);
    CHECK_EQ(toknroll_id(t, "<think>", 7), 151667);
    CHECK_EQ(toknroll_token_flags(t, 151667), TOKNROLL_TOKEN_USER_DEFINED);
    /* one token, four bytes */
    CHECK_EQ(toknroll_encode(t, "\xF0\x9F\x8E\xB8", 4, 0, ids, 64), 1);
    CHECK_EQ(ids[0], 146750);
    toknroll_tokenizer_free(t);
}

/* every special token of a real vocabulary: spelled in a text, it comes back as itself */
static void every_special_token(void) {
    offline();
    toknroll_tokenizer* t = cached(QWEN);
    int control = 0, user = 0, wrong = 0;
    for (int32_t id = 0; id < QWEN_VOCAB; id++) {
        int64_t flags = toknroll_token_flags(t, id);
        if (!(flags & (TOKNROLL_TOKEN_CONTROL | TOKNROLL_TOKEN_USER_DEFINED))) continue;
        control += (flags & TOKNROLL_TOKEN_CONTROL) != 0;
        user += (flags & TOKNROLL_TOKEN_USER_DEFINED) != 0;
        char spelling[64], text[96];
        int64_t n = toknroll_token(t, id, spelling, sizeof spelling);
        int length = snprintf(text, sizeof text, "a%sb", spelling);
        int32_t ids[8];
        int32_t count = toknroll_encode(t, text, (size_t) length, TOKNROLL_SPECIALS, ids, 8);
        if (n <= 0 || count != 3 || ids[1] != id || toknroll_id(t, spelling, (size_t) n) != id) {
            wrong++;
            printf("    %d \"%s\": %d ids\n", (int) id, spelling, (int) count);
        }
    }
    CHECK_EQ(control, 14);
    CHECK_EQ(user, 12);
    CHECK_EQ(wrong, 0);
    toknroll_tokenizer_free(t);
}

static void gguf_reference(void) {   /* a GGUF on the hub: read for its header, from the cache */
    offline();
    toknroll_tokenizer* t = cached(LFM_GGUF);
    CHECK_EQ(toknroll_vocabulary_size(t), 65536);
    static const int32_t hello[] = {36309, 521, 2031, 510};
    int32_t ids[16];
    CHECK_EQ(toknroll_encode(t, "Hello, world!", 13, 0, ids, 16), 4);
    CHECK(same_ids(ids, hello, 4));
    CHECK_EQ(toknroll_id(t, "<|im_end|>", 10), 7);
    toknroll_tokenizer_free(t);
}

/* the engine real vocabularies run on: a batch as encode makes each text, threads as one thread does */
static const char* const PROSE[] = {
    "The quick brown fox jumps over the lazy dog.", "  indented\n\tcode(x) {\r\n return x**2; }",
    "h\xC3\xA9llo w\xC3\xB6rld \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xF0\x9F\x8E\xB8\xF0\x9F\x8E\xB9", "1234567890 3.14159 1e-9",
    "<|im_start|>user\nHi<|im_end|>", "", " ", "\n\n\n", "a", "don't won't it's I'm", "https://example.com/a?b=c#d",
};
enum { PROSE_COUNT = COUNT(PROSE) };

typedef struct { toknroll_tokenizer* t; int32_t (*want)[256]; int32_t* counts; int wrong; } prose_box;
static void prose_worker(void* p) {
    prose_box* b = (prose_box*) p;
    for (int round = 0; round < 200; round++)
        for (int i = 0; i < PROSE_COUNT; i++) {
            int32_t ids[256];
            int32_t n = toknroll_encode(b->t, PROSE[i], strlen(PROSE[i]), TOKNROLL_SPECIALS, ids, 256);
            if (n != b->counts[i] || !same_ids(ids, b->want[i], n)) b->wrong++;
        }
}

static void real_engine(void) {
    offline();
    toknroll_tokenizer* t = cached(QWEN);
    static int32_t want[PROSE_COUNT][256];
    int32_t counts[PROSE_COUNT];
    for (int i = 0; i < PROSE_COUNT; i++) {
        counts[i] = toknroll_encode(t, PROSE[i], strlen(PROSE[i]), TOKNROLL_SPECIALS, want[i], 256);
        char back[512];
        CHECK(counts[i] >= 0);
        CHECK_EQ(toknroll_decode(t, want[i], counts[i], 0, back, sizeof back, NULL), (int64_t) strlen(PROSE[i]));
        CHECK_STR(back, PROSE[i]);
        check_cuts(t, want[i], counts[i], PROSE[i]);
    }
    /* a batch of every text, many times over */
    enum { COPIES = 300 };
    size_t total_bytes = 0;
    for (int i = 0; i < PROSE_COUNT; i++) total_bytes += strlen(PROSE[i]);
    char* texts = (char*) malloc(total_bytes * COPIES + 1);
    int64_t* at = (int64_t*) malloc((PROSE_COUNT * COPIES + 1) * sizeof *at);
    at[0] = 0;
    for (int k = 0; k < PROSE_COUNT * COPIES; k++) {
        size_t n = strlen(PROSE[k % PROSE_COUNT]);
        memcpy(texts + at[k], PROSE[k % PROSE_COUNT], n);
        at[k + 1] = at[k] + (int64_t) n;
    }
    int32_t* offsets = (int32_t*) malloc((PROSE_COUNT * COPIES + 1) * sizeof *offsets);
    int32_t total = toknroll_encode_batch(t, texts, at, PROSE_COUNT * COPIES, TOKNROLL_SPECIALS, NULL, 0, offsets);
    int32_t* out = (int32_t*) malloc((size_t) total * sizeof *out);
    CHECK_EQ(toknroll_encode_batch(t, texts, at, PROSE_COUNT * COPIES, TOKNROLL_SPECIALS, out, total, offsets), total);
    int wrong = 0;
    for (int k = 0; k < PROSE_COUNT * COPIES; k++) {
        int i = k % PROSE_COUNT;
        if (offsets[k + 1] - offsets[k] != counts[i] || !same_ids(out + offsets[k], want[i], counts[i])) wrong++;
    }
    CHECK_EQ(wrong, 0);
    free(out);
    free(offsets);
    free(at);
    free(texts);
    /* eight threads on the one tokenizer, each as the serial pass */
    prose_box boxes[8];
    thread_t running[8];
    for (int i = 0; i < 8; i++) {
        boxes[i] = (prose_box) {t, want, counts, 0};
        running[i] = thread_run(prose_worker, &boxes[i]);
    }
    wrong = 0;
    for (int i = 0; i < 8; i++) {
        thread_join(running[i]);
        wrong += boxes[i].wrong;
    }
    CHECK_EQ(wrong, 0);
    toknroll_tokenizer_free(t);
}

static void gguf_round_trips(void) {
    offline();
    toknroll_tokenizer* t = cached(LFM_GGUF);
    for (int i = 0; i < PROSE_COUNT; i++) {
        int32_t ids[256];
        char back[512];
        int32_t n = toknroll_encode(t, PROSE[i], strlen(PROSE[i]), 0, ids, 256);
        CHECK(n >= 0 && n <= 256);
        CHECK_EQ(toknroll_decode(t, ids, n, 0, back, sizeof back, NULL), (int64_t) strlen(PROSE[i]));
        CHECK_STR(back, PROSE[i]);
    }
    toknroll_tokenizer_free(t);
}

/* the token budget: capacity 0 counts, the same count encode fills, at any size */
static void count_tokens(void) {
    toknroll_tokenizer* t = tiny();
    static const char* texts[] = {"", "a", "hello", "hello world, hello again", "<|end|><|end|>", "\xF0\x9F\x8E\xB8\xE6\x97\xA5"};
    for (int i = 0; i < COUNT(texts); i++) {
        int32_t ids[64];
        size_t n = strlen(texts[i]);
        for (uint32_t flags = 0; flags <= TOKNROLL_SPECIALS; flags++)
            CHECK_EQ(toknroll_encode(t, texts[i], n, flags, NULL, 0), toknroll_encode(t, texts[i], n, flags, ids, 64));
    }
    size_t size = 4u << 20;   /* 4 MiB, counted without a list */
    char* big = (char*) malloc(size);
    for (size_t i = 0; i < size; i++) big[i] = "hello world "[i % 12];
    int32_t counted = toknroll_encode(t, big, size, 0, NULL, 0);
    /* "hello" then " world" and " hello" again and again, " hell" last: 4 + 6, then 5 + 6 a sentence */
    CHECK_EQ(counted, (int32_t) (size / 12) * 11 - 1 + 4);
    int32_t* filled = (int32_t*) malloc((size_t) counted * sizeof *filled);
    CHECK_EQ(toknroll_encode(t, big, size, 0, filled, counted), counted);
    free(filled);
    free(big);
    toknroll_tokenizer_free(t);
}

/* the idiom every binding writes: ask the size, allocate it, fill it; whole exactly when below capacity */
static void size_then_fill(void) {
    toknroll_tokenizer* t = tiny();
    const char* text = "h\xC3\xA9llo <|end|> w\xC3\xB6rld";
    size_t length = strlen(text);
    int32_t n = toknroll_encode(t, text, length, TOKNROLL_SPECIALS, NULL, 0);
    int32_t* ids = (int32_t*) malloc((size_t) n * sizeof *ids);
    CHECK_EQ(toknroll_encode(t, text, length, TOKNROLL_SPECIALS, ids, n), n);
    int64_t bytes = toknroll_decode(t, ids, n, 0, NULL, 0, NULL);
    CHECK_EQ(bytes, (int64_t) length);
    char* out = (char*) malloc((size_t) bytes + 1);
    CHECK_EQ(toknroll_decode(t, ids, n, 0, out, bytes + 1, NULL), bytes);   /* below capacity: whole */
    CHECK(!memcmp(out, text, length + 1));
    CHECK_EQ(toknroll_decode(t, ids, n, 0, out, bytes, NULL), bytes);       /* at capacity: cut, the NUL in the last byte */
    CHECK((int64_t) strlen(out) < bytes);
    int64_t spelled = toknroll_token(t, ID_END, NULL, 0);
    char token[16];
    CHECK_EQ(toknroll_token(t, ID_END, token, spelled + 1), spelled);
    CHECK_STR(token, "<|end|>");
    free(out);
    free(ids);
    toknroll_tokenizer_free(t);
}

/* a reply decoded as it streams, token by token: the bytes add up to the whole, half characters and all */
static void streaming_decode(void) {
    toknroll_tokenizer* t = tiny();
    const char* text = "Tok'n'Roll \xF0\x9F\x8E\xB8 \xE6\x97\xA5\xE6\x9C\xAC <|end|>";
    int32_t ids[128];
    int32_t n = toknroll_encode(t, text, strlen(text), TOKNROLL_SPECIALS, ids, 128);
    char streamed[256] = "";
    size_t at = 0;
    int half = 0;
    for (int32_t i = 0; i < n; i++) {
        char piece[16];
        int64_t k = toknroll_decode(t, ids + i, 1, 0, piece, sizeof piece, NULL);
        CHECK(k > 0 && k < 16);
        memcpy(streamed + at, piece, (size_t) k);
        at += (size_t) k;
        half += k == 1 && ((unsigned char) piece[0] & 0x80);   /* a byte of a longer character */
    }
    streamed[at] = 0;
    CHECK_STR(streamed, text);
    CHECK(half >= 10);   /* the emoji and the two kanji came a byte at a time */
    toknroll_tokenizer_free(t);
}

/* an added token is spelled as its text, the one TOKNROLL_SPECIALS matches: phi-2's 50257 is 31 spaces,
 * which its byte-level vocabulary writes "ĠĠĠ..." */
static void added_token_text(void) {
    offline();
    toknroll_tokenizer* t = cached("microsoft/phi-2");
    char text[64];
    CHECK_EQ(toknroll_token(t, 50257, text, sizeof text), 31);
    CHECK(strspn(text, " ") == 31);
    CHECK_EQ(toknroll_id(t, text, 31), 50257);
    int32_t ids[64];
    CHECK_EQ(toknroll_encode(t, text, 31, TOKNROLL_SPECIALS, ids, 64), 1);
    CHECK_EQ(ids[0], 50257);
    toknroll_tokenizer_free(t);
}

/* every id's spelling leads back to it; every id has exactly one type and no reserved bit */
static void vocabulary_round_trip(void) {
    toknroll_tokenizer* t = tiny();
    int wrong = 0;
    for (int32_t id = 0; id < VOCAB_JSON; id++) {
        char spelling[32];
        int64_t n = toknroll_token(t, id, spelling, sizeof spelling);
        int64_t flags = toknroll_token_flags(t, id);
        if (n <= 0 || toknroll_id(t, spelling, (size_t) n) != id || !one_type(flags) || (flags & ~INT64_C(0x7E))) wrong++;
    }
    CHECK_EQ(wrong, 0);
    toknroll_tokenizer_free(t);
}

/* the README's recipe: a text without its special tokens, by their flags */
static void without_specials(void) {
    toknroll_tokenizer* t = tiny();
    const char* text = "<|end|>hello<user> world<|end|>";
    int32_t ids[64], kept[64], k = 0;
    int32_t n = toknroll_encode(t, text, strlen(text), TOKNROLL_SPECIALS, ids, 64);
    for (int32_t i = 0; i < n; i++)
        if (toknroll_token_flags(t, ids[i]) & TOKNROLL_TOKEN_NORMAL) kept[k++] = ids[i];
    char out[64];
    CHECK_EQ(toknroll_decode(t, kept, k, 0, out, 64, NULL), 11);
    CHECK_STR(out, "hello world");
    toknroll_tokenizer_free(t);
}

/* ---- odd arguments ---- */

static void odd_arguments(void) {
    toknroll_tokenizer* t = tiny();
    int32_t ids[300];
    CHECK_ERROR(toknroll_encode(t, "x", 1, 0xFFFFFFFFu, ids, 8), TOKNROLL_EINVAL, LE, "unknown flags 0xfffffffe");
    /* a capacity far past the buffer writes only what there is */
    for (int i = 0; i < 8; i++) ids[i] = -7;
    CHECK_EQ(toknroll_encode(t, "hello", 5, 0, ids, INT32_MAX), 4);
    CHECK_EQ(ids[4], -7);
    char out[32];
    memset(out, 'Z', sizeof out);
    CHECK_EQ(toknroll_decode(t, HELLO, 4, 0, out, INT64_MAX, NULL), 5);
    CHECK(!strcmp(out, "hello") && out[6] == 'Z');
    CHECK_EQ(toknroll_token(t, ID_HE, out, INT64_MAX), 2);
    /* a spelling with a NUL inside, or bytes that are no UTF-8: no such token, never a crash */
    CHECK_ERROR(toknroll_id(t, "he\0", 3), TOKNROLL_ENOTFOUND, LE, "no token");
    CHECK_ERROR(toknroll_id(t, "\xFF\xFE", 2), TOKNROLL_ENOTFOUND, LE, "no token");
    CHECK_ERROR(toknroll_id(t, "\xC4", 1), TOKNROLL_ENOTFOUND, LE, "no token");   /* half of "Ġ" */
    /* every ASCII byte, NUL and controls included, comes back as it went in */
    unsigned char ascii[128];
    for (int b = 0; b < 128; b++) ascii[b] = (unsigned char) b;
    int32_t n = toknroll_encode(t, (const char*) ascii, 128, 0, ids, 300);
    CHECK_EQ(n, 128);
    char back[300];
    CHECK_EQ(toknroll_decode(t, ids, n, 0, back, sizeof back, NULL), 128);
    CHECK(!memcmp(back, ascii, 128));
    /* a malformed sequence reads as U+FFFD (EF BF BD), one per bad sequence, as a UTF-8 decoder reads it */
    static const struct { const char* bad; const char* read; } malformed[] = {
        {"\xFF", "\xEF\xBF\xBD"},                        /* never a UTF-8 byte */
        {"\x80", "\xEF\xBF\xBD"},                        /* a lone continuation */
        {"a\xC3", "a\xEF\xBF\xBD"},                      /* cut short at the end */
        {"\xC3(", "\xEF\xBF\xBD("},                      /* cut short by an ASCII byte */
        {"\xC0\xAF", "\xEF\xBF\xBD\xEF\xBF\xBD"},    /* overlong */
        {"\xED\xA0\x80", "\xEF\xBF\xBD"},              /* a surrogate: one, as Java's decoder reads it (Python's: three) */
    };
    for (int i = 0; i < COUNT(malformed); i++) {
        int32_t got[16], want[16];
        int32_t g = toknroll_encode(t, malformed[i].bad, strlen(malformed[i].bad), 0, got, 16);
        int32_t w = toknroll_encode(t, malformed[i].read, strlen(malformed[i].read), 0, want, 16);
        CHECK(g == w && same_ids(got, want, w));
        if (g != w) printf("    malformed %d: %d ids, U+FFFD reading %d\n", i, (int) g, (int) w);
    }
    /* only whitespace; a thousand specials in a row */
    CHECK_EQ(toknroll_encode(t, " \t\r\n ", 5, 0, ids, 300), 5);
    static char ends[7001];
    for (int i = 0; i < 1000; i++) memcpy(ends + 7 * i, "<|end|>", 7);
    static int32_t many[1000];
    CHECK_EQ(toknroll_encode(t, ends, 7000, TOKNROLL_SPECIALS, many, 1000), 1000);
    int all_end = 1;
    for (int i = 0; i < 1000; i++) all_end &= many[i] == ID_END;
    CHECK(all_end);
    /* one long run of one letter */
    size_t run = 200000;
    char* as = (char*) malloc(run);
    memset(as, 'a', run);
    int32_t count = toknroll_encode(t, as, run, 0, NULL, 0);
    CHECK_EQ(count, (int32_t) run);
    free(as);
    toknroll_tokenizer_free(t);
}

static void odd_batches(void) {
    toknroll_tokenizer* t = tiny();
    /* texts with NULs inside, and empty ones at either end */
    const char texts[] = "a\0b" "hello";
    const int64_t at[] = {0, 0, 3, 8, 8};
    int32_t out[16], offsets[5];
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 4, 0, out, 16, offsets), 7);
    CHECK(offsets[1] == 0 && offsets[2] == 3 && offsets[3] == 7 && offsets[4] == 7);
    CHECK(out[0] == 'a' && out[1] == 0 && out[2] == 'b' && out[3] == ID_HE);
    /* capacity exactly the total, and one short of it */
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 4, 0, out, 7, offsets), 7);
    out[6] = -7;
    CHECK_EQ(toknroll_encode_batch(t, texts, at, 4, 0, out, 6, offsets), 7);
    CHECK_EQ(out[6], -7);
    /* one text, alone in the batch, as encode makes it */
    static char one[100001];
    for (int i = 0; i < 100000; i++) one[i] = "hello "[i % 6];
    const int64_t whole[] = {0, 100000};
    int32_t total = toknroll_encode_batch(t, one, whole, 1, 0, NULL, 0, offsets);
    CHECK_EQ(total, toknroll_encode(t, one, 100000, 0, NULL, 0));
    CHECK(offsets[0] == 0 && offsets[1] == total);
    toknroll_tokenizer_free(t);
}

/* ---- many tokenizers, many lifetimes ---- */

static void many_tokenizers(void) {
    const char* gguf_path = TMP("many.gguf");
    write_tiny_gguf(gguf_path, "gpt2");
    toknroll_tokenizer* a = tiny();
    toknroll_tokenizer* b = toknroll_tokenizer_load(gguf_path);
    toknroll_tokenizer* c = tiny();
    CHECK(a && b && c && a != c);
    int32_t ids[16];
    for (int round = 0; round < 50; round++) {   /* interleaved, each its own answers */
        CHECK_EQ(toknroll_vocabulary_size(a), VOCAB_JSON);
        CHECK_EQ(toknroll_vocabulary_size(b), VOCAB_GGUF);
        CHECK_EQ(toknroll_token_flags(b, ID_UNUSED), TOKNROLL_TOKEN_UNUSED);
        CHECK_EQ(toknroll_token_flags(a, ID_UNUSED), TOKNROLL_ENOTFOUND);
        CHECK_EQ(toknroll_encode(c, "hello", 5, 0, ids, 16), 4);
    }
    toknroll_tokenizer_free(b);   /* any order */
    CHECK_EQ(toknroll_encode(a, "hello", 5, 0, ids, 16), 4);
    toknroll_tokenizer_free(a);
    CHECK_EQ(toknroll_encode(c, "hello", 5, 0, ids, 16), 4);
    toknroll_tokenizer_free(c);
}

#ifdef __linux__
static long resident_kb(void) {
    FILE* f = fopen("/proc/self/statm", "r");
    long pages = 0, resident = 0;
    if (f && fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
    if (f) fclose(f);
    return resident * (sysconf(_SC_PAGESIZE) / 1024);
}
#endif

static void load_free_cycles(void) {   /* a freed tokenizer leaves nothing behind that grows */
    const char* path = tiny_json();
    toknroll_tokenizer_free(toknroll_tokenizer_load(path));
#ifdef __linux__
    long before = resident_kb();
#endif
    int failed = 0;
    for (int i = 0; i < 300; i++) {
        toknroll_tokenizer* t = toknroll_tokenizer_load(path);
        int32_t ids[8];
        failed += !t || toknroll_encode(t, "hello", 5, 0, ids, 8) != 4;
        toknroll_tokenizer_free(t);
    }
    CHECK_EQ(failed, 0);
#ifdef __linux__
    long grown = resident_kb() - before;
    if (grown > 256 * 1024) printf("    resident memory grew %ld MB over 300 loads\n", grown / 1024);
    CHECK(grown < 256 * 1024);
#endif
}

/* ---- configure, the finer points ---- */

static void configure_value_with_spaces(void) {   /* one option is one string: spaces and all */
    const char* regex[] = {"-Dtoknroll.gguf.pre.ctest-pre=regex: ?[a-z]+|\\s+"};
    CHECK_EQ(toknroll_configure(regex, 1), TOKNROLL_OK);
    toknroll_tokenizer* t = toknroll_tokenizer_load(custom_gguf());
    CHECK(t != NULL);
    int32_t ids[16];
    static const int32_t want[] = {ID_HE, 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'};   /* " world" split off whole */
    CHECK_EQ(toknroll_encode(t, "hello world", 11, 0, ids, 16), 10);
    CHECK(same_ids(ids, want, 10));
    toknroll_tokenizer_free(t);
}

static void configure_many_options(void) {
    static char storage[64][48];
    const char* options[64];
    for (int i = 0; i < 64; i++) {
        snprintf(storage[i], sizeof storage[i], "-Dctest.option.%d=value %d", i, i);
        options[i] = storage[i];
    }
    CHECK_EQ(toknroll_configure(options, 64), TOKNROLL_OK);
    toknroll_tokenizer_free(tiny());
    CHECK_CLEAR(LE);
}

/* ---- the strings the library hands out ---- */

static void returned_strings(void) {
    CHECK(toknroll_version() == toknroll_version());   /* static: the same pointer every time */
    toknroll_tokenizer* t = tiny();
    toknroll_token(t, -1, NULL, 0);
    const char* message = LE(NULL);
    char copy[128];
    snprintf(copy, sizeof copy, "%s", message);
    toknroll_version();
    CHECK_STR(message, copy);                          /* valid until the thread's next call */
    CHECK(strlen(message) > 0 && strlen(message) < 1024);
    /* every refusal says something, and its code is the call's own */
    int32_t code = 0;
    toknroll_encode(t, NULL, 1, 0, NULL, 0);
    CHECK(LE(&code)[0] != 0 && code == TOKNROLL_EINVAL);
    toknroll_id(t, "nope", 4);
    CHECK(LE(&code)[0] != 0 && code == TOKNROLL_ENOTFOUND);
    toknroll_tokenizer_free(t);
}

static const scenario scenarios[] = {
    {"version", version, NULL},
    {"last_error_per_call", last_error_per_call, NULL},
    {"last_error_before_start", last_error_before_start, NULL},
    {"last_error_per_thread", last_error_per_thread, NULL},
    {"vocabulary_with_a_gap", vocabulary_with_a_gap, NULL},
    {"configure_arguments", configure_arguments, NULL},
    {"configure_after_a_call", configure_after_a_call, NULL},
    {"configure_rejected", configure_rejected, NULL},
    {"configure_ignores_unknown_x", configure_ignores_unknown_x, NULL},
    {"options_unknown_pre", options_unknown_pre, NULL},
    {"options_take_effect", options_take_effect, NULL},
    {"options_from_the_environment", options_from_the_environment, OPTIONS_VAR "=" ALIAS_CTEST},
    {"options_utf8", options_utf8, NULL},
    {"options_from_the_environment_utf8", options_from_the_environment_utf8,
     OPTIONS_VAR "=-Dtoknroll.gguf.pre.caf\xC3\xA9=alias:gpt2"},
    {"options_environment_wins", options_environment_wins, OPTIONS_VAR "=-Dtoknroll.gguf.pre.ctest-pre=alias:no-such-scheme"},
    {"options_environment_rejected", options_environment_rejected, OPTIONS_VAR "= -XX:+NoSuchOptionAnywhere "},
    {"options_rejected_either_way", options_rejected_either_way, OPTIONS_VAR "=-XX:+NoSuchOptionAnywhere"},
    {"log_set_once", log_set_once, NULL},
    {"writes_nothing_to_stdout", writes_nothing_to_stdout, NULL},
    {"log_set_after_a_call", log_set_after_a_call, NULL},
    {"log_set_after_configure", log_set_after_configure, NULL},
    {"log_to_stderr_by_default", log_to_stderr_by_default, NULL},
    {"log_to_the_callback", log_to_the_callback, NULL},
    {"log_null_is_silence", log_null_is_silence, LOG_VAR "=debug"},
    {"log_off", log_filtered, LOG_VAR "=off"},
    {"log_error_only", log_filtered, LOG_VAR "=error"},
    {"log_warn", log_passed, LOG_VAR "=warn"},
    {"log_debug", log_passed, LOG_VAR "=debug"},
    {"log_any_case", log_filtered, LOG_VAR "= Error "},
    {"log_empty_is_unset", log_passed, LOG_VAR "="},
    {"log_unknown_value", log_unknown_value, LOG_VAR "=verbose"},
    {"log_off_on_stderr", stderr_quiet, LOG_VAR "=off"},
    {"log_error_on_stderr", stderr_quiet, LOG_VAR "=error"},
    {"log_callback_calls_in", log_callback_calls_in, LOG_VAR "=verbose"},
    {"log_unknown_value_on_stderr", log_unknown_value_on_stderr, LOG_VAR "=verbose"},
    {"load_paths", load_paths, NULL},
    {"load_folders", load_folders, NULL},
    {"load_by_content", load_by_content, NULL},
    {"load_references", load_references, NULL},
    {"load_unsupported", load_unsupported, NULL},
    {"load_from_the_hub_cache", load_from_the_hub_cache, "HF_HUB_CACHE=" TMP("hfhub")},
    {"load_cache_root_from_the_environment", load_cache_root_from_the_environment,
     "TOKNROLL_CACHE_ROOT=" TMP("envcache")},
    {"encode", encode, NULL},
    {"encode_specials", encode_specials, NULL},
    {"error_messages", error_messages, NULL},
    {"decode", decode, NULL},
    {"decode_cuts", decode_cuts, NULL},
    {"decode_resumes", decode_resumes, NULL},
    {"round_trips", round_trips, NULL},
    {"long_text", long_text, NULL},
    {"decode_a_token_past_the_chunk", decode_a_token_past_the_chunk, NULL},
    {"batch", batch, NULL},
#ifdef __linux__
    {"batch_threads_capped", batch_threads_capped, OPTIONS_VAR "=-XX:ActiveProcessorCount=2"},
#endif
    {"batch_matches_encode", batch_matches_encode, NULL},
    {"vocabulary", vocabulary, NULL},
    {"token_flags", token_flags, NULL},
    {"gguf", gguf, NULL},
    {"threads", threads, NULL},
    {"hub_reference", hub_reference, NULL},
#ifdef __linux__
    {"address_space_too_small", address_space_too_small, NULL},
#endif
    {"heap_too_small", heap_too_small, NULL},
    PROGRESS_SCENARIOS
    {"every_special_token", every_special_token, NULL},
    {"gguf_reference", gguf_reference, NULL},
    {"real_engine", real_engine, NULL},
    {"gguf_round_trips", gguf_round_trips, NULL},
    {"count_tokens", count_tokens, NULL},
    {"size_then_fill", size_then_fill, NULL},
    {"streaming_decode", streaming_decode, NULL},
    {"added_token_text", added_token_text, NULL},
    {"vocabulary_round_trip", vocabulary_round_trip, NULL},
    {"without_specials", without_specials, NULL},
    {"odd_arguments", odd_arguments, NULL},
    {"odd_batches", odd_batches, NULL},
    {"many_tokenizers", many_tokenizers, NULL},
    {"load_free_cycles", load_free_cycles, NULL},
    {"configure_value_with_spaces", configure_value_with_spaces, NULL},
    {"options_environment_whitespace", options_from_the_environment, OPTIONS_VAR "=\t-Dtoknroll.offline=true\n   " ALIAS_CTEST " \r\n"},
    {"options_environment_empty", options_take_effect, OPTIONS_VAR "= \t \n"},
    {"configure_many_options", configure_many_options, NULL},
    {"returned_strings", returned_strings, NULL},
#ifndef _WIN32
    {"fork_children", fork_children, NULL},
    {"fork_during_first_batch", fork_during_first_batch, NULL},
    {"load_utf8_path", load_utf8_path, NULL},
    {"load_fifo", load_fifo, NULL},
#endif
};

int main(int argc, char** argv) {
    return test_main(argc, argv, scenarios, COUNT(scenarios));
}
