/* libjinfer's C API, function by function: every documented result, error code and special case, on
 * the models the hub cache holds, offline. A language and an embedding model are required; the
 * reranker, speech and transcription checks run when their model is cached, and say so when skipped.
 * toknroll's own tests run against the toknroll_* functions this library carries (TEST_LIBJINFER).
 *
 *   ./jinfer_test           every scenario, each in a fresh process
 *   ./jinfer_test <name>    one, here */
#include "test.h"   /* first: it picks the POSIX features */
#include "jinfer.h"

#define LM "LiquidAI/LFM2.5-350M-GGUF:Q8_0"
#define EMB "LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0"
#define RERANKER "mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0"
#define SPEAKER "simonfxr/kokoro.cpp-GGUF:Q8_0"
#define VOICE "voice=simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf"
#define TRANSCRIBER "mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf"
#define HI "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One word.\"}],\"temperature\":0,\"max_tokens\":8}"
#define STORY "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me a long story about a lighthouse.\"}],\"temperature\":0,\"max_tokens\":400}"

#define LE jinfer_last_error

static void offline(void) {
    const char* options[] = {"-Djinfer.offline=true"};   /* a test never downloads */
    if (jinfer_configure(options, 1) != JINFER_OK) { printf("    configure: %s\n", LE(NULL)); exit(1); }
}

static jinfer_model* load(const char* ref) {
    jinfer_model* m = jinfer_model_load(ref, NULL);
    if (!m) { printf("    %s does not load (not cached?): %s\n", ref, LE(NULL)); exit(1); }
    return m;
}

/* a model the scenario needs from the cache: skipped, and counted so, when it is not there */
static jinfer_model* cached_model(const char* ref, const char* companion) {
    jinfer_model_options o;
    jinfer_model_options_init(&o);
    const char* companions[] = {companion};
    o.companions = companion ? companions : NULL;
    o.companion_count = companion ? 1 : 0;
    jinfer_model* m = jinfer_model_load(ref, &o);
    if (!m) skip("%s is not cached (%s)", ref, LE(NULL));
    return m;
}

/* the ids of text, through the model's own tokenizer */
static int32_t ids_of(jinfer_model* m, const char* text, int32_t* out, int32_t capacity) {
    toknroll_tokenizer* t = jinfer_model_tokenizer(m);
    int32_t n = toknroll_encode(t, text, strlen(text), 0, out, capacity);
    toknroll_tokenizer_free(t);
    return n;
}

/* piece after text[0, *at), as much as fits */
static void append(char* text, size_t size, size_t* at, const char* piece) {
    int n = snprintf(text + *at, size - *at, "%s", piece);
    if (n > 0) *at += (size_t) n < size - *at ? (size_t) n : size - *at - 1;
}

/* a whole stream: its text, and the step's last return (DONE, or what stopped it) */
static int32_t drain(jinfer_stream* s, char* text, size_t size) {
    size_t at = 0;
    int32_t rc;
    text[0] = 0;
    while ((rc = jinfer_stream_step(s)) == JINFER_OK) append(text, size, &at, jinfer_stream_text(s));
    return rc;
}

/* ---- version, last error, settings ---- */

static void version(void) {
    char want[32];
    snprintf(want, sizeof want, "%d.%d.%d", JINFER_VERSION_MAJOR, JINFER_VERSION_MINOR, JINFER_VERSION_PATCH);
    CHECK_STR(jinfer_version(), want);
    snprintf(want, sizeof want, "%d.%d.%d", TOKNROLL_VERSION_MAJOR, TOKNROLL_VERSION_MINOR, TOKNROLL_VERSION_PATCH);
    CHECK_STR(toknroll_version(), want);   /* the toknroll it carries */
    CHECK_CLEAR(LE);
    CHECK_STR(LE(NULL), "");
    jinfer_model_options o;
    memset(&o, 0xAB, sizeof o);
    jinfer_model_options_init(&o);
    CHECK_EQ(o.struct_size, sizeof o);
    CHECK(o.companions == NULL && o.companion_count == 0 && o.prompt_cache == NULL && o.context_capacity == 0);
    /* frees of NULL, and the calls above, start nothing: configure still comes first */
    jinfer_model_free(NULL);
    jinfer_state_free(NULL);
    jinfer_stream_free(NULL);
    jinfer_stream_cancel(NULL);
    CHECK_EQ(jinfer_configure(NULL, 0), JINFER_OK);
}

static void configure(void) {
    const char* one[] = {"-Djinfer.offline=true", NULL};
    CHECK_ERROR(jinfer_configure(NULL, 1), JINFER_EINVAL, LE, "jinfer_configure: options is NULL or count negative");
    CHECK_ERROR(jinfer_configure(one, -1), JINFER_EINVAL, LE, "options is NULL or count negative");
    CHECK_ERROR(jinfer_configure(one, 2), JINFER_EINVAL, LE, "jinfer_configure: options[1] is NULL");
    const char* blank[] = {""};
    CHECK_ERROR(jinfer_configure(blank, 1), JINFER_EINVAL, LE, "jinfer_configure: options[0] is blank");
    CHECK_EQ(jinfer_configure(one, 1), JINFER_OK);
    CHECK_CLEAR(LE);
    CHECK_ERROR(jinfer_configure(one, 1), JINFER_EINVAL, LE, "jinfer_configure must come once, before any other call into the library");
    /* toknroll_configure is the same call on the same runtime */
    CHECK_ERROR(toknroll_configure(one, 1), TOKNROLL_EINVAL, toknroll_last_error, "toknroll_configure must come once");
    CHECK_ERROR(jinfer_log_set(NULL, NULL), JINFER_EINVAL, LE, "jinfer_log_set must come once");
}

static void start_the_runtime(void* p) {
    (void) p;
    jinfer_model_property(NULL, JINFER_PROPERTY_VOCAB_SIZE);   /* refused, on this thread alone */
}

/* a refusal before the runtime starts stays the failing thread's, once another thread starts it, until
 * that thread's own next call */
static void last_error_before_start(void) {
    CHECK_ERROR(jinfer_configure(NULL, 1), JINFER_EINVAL, LE, "options is NULL");
    thread_join(thread_run(start_the_runtime, NULL));
    CHECK_ERROR(JINFER_EINVAL, JINFER_EINVAL, LE, "jinfer_configure: options is NULL");
    CHECK_ERROR(jinfer_model_property(NULL, JINFER_PROPERTY_VOCAB_SIZE), JINFER_EINVAL, LE, "model is NULL");
}

static void error_channels(void) {
    CHECK_ERROR(toknroll_vocabulary_size(NULL), TOKNROLL_EINVAL, toknroll_last_error, "tokenizer is NULL");
    CHECK_STR(jinfer_model_info(NULL), "");   /* returning an empty result must preserve the error */
    CHECK_ERROR(JINFER_EINVAL, JINFER_EINVAL, LE, "model is NULL");
    CHECK_ERROR(TOKNROLL_EINVAL, TOKNROLL_EINVAL, toknroll_last_error, "tokenizer is NULL");
    CHECK_ERROR(toknroll_decode(NULL, NULL, 0, 0, NULL, 0, NULL), TOKNROLL_EINVAL, toknroll_last_error, "tokenizer is NULL");
    CHECK_ERROR(JINFER_EINVAL, JINFER_EINVAL, LE, "model is NULL");
}

static void configure_through_toknroll(void) {
    const char* one[] = {"-Djinfer.offline=true"};
    CHECK_EQ(toknroll_configure(one, 1), TOKNROLL_OK);
    CHECK_ERROR(jinfer_configure(one, 1), JINFER_EINVAL, LE, "jinfer_configure must come once");
    CHECK_NULL_ERROR(jinfer_model_load("ctest-nobody/no-such-GGUF:Q8_0", NULL), JINFER_ENOTFOUND, LE, "not cached");   /* offline held */
}

static void configure_rejected(void) {
    const char* bad[] = {"-XX:+NoSuchOptionAnywhere"};
    capture_stderr();
    CHECK_ERROR(jinfer_configure(bad, 1), JINFER_ERROR, LE, "an option from jinfer_configure was rejected, see stderr for which");
    char* err = captured_stderr();
    CHECK_HAS(err, "NoSuchOptionAnywhere");
    free(err);
    CHECK_NULL_ERROR(jinfer_model_load(LM, NULL), JINFER_ERROR, LE, "the runtime could not be started");
    CHECK_ERROR(jinfer_model_property(NULL, JINFER_PROPERTY_VOCAB_SIZE), JINFER_ERROR, LE, "could not be started");
    CHECK_STR(jinfer_model_info(NULL), "");
    CHECK_ERROR(jinfer_configure(NULL, 0), JINFER_EINVAL, LE, "jinfer_configure must come once");
}

static void options_from_the_environment(void) {   /* JINFER_OPTIONS=-Djinfer.threads=3, over configure's 5 */
    const char* five[] = {"-Djinfer.offline=true", "-Djinfer.threads=5"};
    CHECK_EQ(jinfer_configure(five, 2), JINFER_OK);
    jinfer_model* m = load(LM);
    CHECK_HAS(jinfer_model_info(m), "\"threads\":3");
    jinfer_model_free(m);
}

/* ---- logging ---- */

typedef struct { int lines, debug, other, broken, traces; char first[512]; void* user_data; } log_box;
static log_box seen;
static void keep_line(int32_t level, const char* message, void* user_data) {
    if (!seen.lines++) snprintf(seen.first, sizeof seen.first, "%s", message);
    seen.broken += strchr(message, '\n') || strchr(message, '\r');   /* a line, never more */
    seen.traces += strstr(message, " | at ") != NULL;                /* a stack trace, folded */
    if (level == JINFER_LOG_DEBUG) seen.debug++; else seen.other++;
    seen.user_data = user_data;
}

static void log_set_once(void) {
    CHECK_EQ(jinfer_log_set(keep_line, &seen), JINFER_OK);
    CHECK_CLEAR(LE);
    CHECK_ERROR(jinfer_log_set(keep_line, &seen), JINFER_EINVAL, LE, "jinfer_log_set must come once, before any other call into the library");
    /* toknroll_log_set is the same sink: set already */
    CHECK_ERROR(toknroll_log_set(keep_line, &seen), TOKNROLL_EINVAL, toknroll_last_error, "toknroll_log_set must come once");
    offline();   /* log_set starts nothing: configure follows */
    CHECK_ERROR(jinfer_log_set(keep_line, &seen), JINFER_EINVAL, LE, "jinfer_log_set must come once");
}

static void log_set_through_toknroll(void) {
    CHECK_EQ(toknroll_log_set(keep_line, &seen), TOKNROLL_OK);
    CHECK_ERROR(jinfer_log_set(keep_line, &seen), JINFER_EINVAL, LE, "jinfer_log_set must come once");
}

static void log_set_after_a_call(void) {
    offline();
    CHECK_ERROR(jinfer_log_set(keep_line, NULL), JINFER_EINVAL, LE, "jinfer_log_set must come once");
}

static void log_quiet_by_default(void) {   /* a load and a chat say nothing at warn */
    capture_stderr();
    CHECK_EQ(jinfer_log_set(keep_line, NULL), JINFER_OK);
    offline();
    jinfer_model* m = load(LM);
    char text[256];
    jinfer_stream* s = jinfer_chat(m, HI);
    CHECK_EQ(drain(s, text, sizeof text), JINFER_DONE);
    jinfer_stream_free(s);
    jinfer_model_free(m);
    CHECK_EQ(seen.lines, 0);
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void log_quiet_on_stderr(void) {   /* no callback: a load and a chat print nothing either */
    capture_stderr();
    offline();
    jinfer_model* m = load(LM);
    char text[256];
    jinfer_stream* s = jinfer_chat(m, HI);
    CHECK_EQ(drain(s, text, sizeof text), JINFER_DONE);
    jinfer_stream_free(s);
    jinfer_model_free(m);
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

/* the start's own warning (an unknown JINFER_LOG) reaches the callback from inside the start, and the
 * callback may call in: it must not hang there */
static int called_in;
static void call_in(int32_t level, const char* message, void* user_data) {
    (void) level; (void) message; (void) user_data;
    jinfer_last_error(NULL);
    toknroll_last_error(NULL);
    called_in++;
}

static void log_callback_calls_in_at_start(void) {   /* JINFER_LOG=loud */
#ifndef _WIN32
    alarm(60);
#endif
    CHECK_EQ(jinfer_log_set(call_in, NULL), JINFER_OK);
    offline();
    CHECK_EQ(called_in, 1);
    CHECK_EQ(jinfer_model_property(NULL, JINFER_PROPERTY_VOCAB_SIZE), JINFER_EINVAL);   /* and serves on */
}

static void log_debug_to_the_callback(void) {   /* JINFER_LOG=debug */
    static int tag;
    capture_stderr();
    CHECK_EQ(jinfer_log_set(keep_line, &tag), JINFER_OK);
    offline();
    jinfer_model* m = load(LM);
    CHECK(seen.debug > 0 && seen.user_data == &tag);
    CHECK_HAS(seen.first, "jinfer.native: ");
    CHECK(!strchr(seen.first, '\n'));
    int before = seen.lines;
    /* a failure's stack trace comes at debug, before the error is kept */
    CHECK_ERROR(jinfer_model_property(m, 999), JINFER_EINVAL, LE, "unknown jinfer_model_property key 999");
    CHECK(seen.lines > before);
    CHECK(seen.traces > 0);
    CHECK_EQ(seen.broken, 0);
    jinfer_model_free(m);
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void log_debug_to_stderr(void) {   /* JINFER_LOG=debug, no callback */
    capture_stderr();
    offline();
    jinfer_model* m = load(LM);
    jinfer_model_property(m, 999);   /* a stack trace, on one line too */
    jinfer_model_free(m);
    char* err = captured_stderr();
    int unprefixed = 0;
    for (const char* line = err; line && *line; line = strstr(line, NL) ? strstr(line, NL) + strlen(NL) : "")
        unprefixed += strncmp(line, "jinfer: ", 8) != 0;
    CHECK_EQ(unprefixed, 0);         /* every line its own, prefixed */
    CHECK_HAS(err, " | at ");
    CHECK_HAS(err, "jinfer: debug: jinfer.native: loaded language LFM2.5-350M-Q8_0.gguf" NL);
    CHECK_HAS(err, "jinfer: debug: jinfer.native: freed LFM2.5-350M-Q8_0.gguf" NL);
    free(err);
}

static void log_null_is_silence(void) {   /* JINFER_LOG=debug, the callback NULL: nothing */
    capture_stderr();
    CHECK_EQ(jinfer_log_set(NULL, NULL), JINFER_OK);
    offline();
    jinfer_model* m = load(LM);
    jinfer_model_property(m, 999);
    jinfer_model_free(m);
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void log_off(void) {   /* JINFER_LOG=off */
    capture_stderr();
    CHECK_EQ(jinfer_log_set(keep_line, NULL), JINFER_OK);
    offline();
    jinfer_model* m = load(LM);
    jinfer_model_property(m, 999);
    jinfer_model_free(m);
    CHECK_EQ(seen.lines, 0);
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void log_unknown_value(void) {   /* JINFER_LOG=loud */
    capture_stderr();
    offline();
    char* err = captured_stderr();
    CHECK_STR(err, "jinfer: warn: jinfer.native: JINFER_LOG=loud is not off, error, warn, info or debug: warn applies" NL);
    free(err);
}

static void log_toknroll_log_is_not_read(void) {   /* TOKNROLL_LOG=bogus means nothing to libjinfer */
    capture_stderr();
    offline();
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

/* the callback may call jinfer_last_error and the _free functions: here it frees a model mid-failure */
static jinfer_model* spare;
static int freed_inside;
static void free_inside(int32_t level, const char* message, void* user_data) {
    (void) level; (void) message; (void) user_data;
    jinfer_model* m = spare;
    spare = NULL;   /* first: the free logs too, and that line comes back here */
    if (m) {
        jinfer_last_error(NULL);
        jinfer_model_free(m);
        freed_inside = 1;
    }
}

static void log_callback_may_free(void) {   /* JINFER_LOG=debug */
    CHECK_EQ(jinfer_log_set(free_inside, NULL), JINFER_OK);
    offline();
    jinfer_model* m = load(LM);
    spare = load(LM);
    CHECK_ERROR(jinfer_model_property(m, 999), JINFER_EINVAL, LE, "unknown jinfer_model_property key 999");   /* the error kept */
    CHECK(freed_inside && spare == NULL);
    jinfer_model_free(m);
}

/* ---- loading ---- */

static void load_errors(void) {
    offline();
    CHECK_NULL_ERROR(jinfer_model_load(NULL, NULL), JINFER_EINVAL, LE, "source is NULL");
    CHECK_NULL_ERROR(jinfer_model_load("", NULL), JINFER_EINVAL, LE, "source is empty");
    CHECK_NULL_ERROR(jinfer_model_load("./" TMP("missing.gguf"), NULL), JINFER_ENOTFOUND, LE, "no such file");
    CHECK_NULL_ERROR(jinfer_model_load("./ctest-missing.gguf", NULL), JINFER_ENOTFOUND, LE, "no such file");
    CHECK_NULL_ERROR(jinfer_model_load("~/ctest-missing.gguf", NULL), JINFER_ENOTFOUND, LE, "no such file");
    /* owner/repo/file is a reference by the grammar, a file in a repository: not cached */
    CHECK_NULL_ERROR(jinfer_model_load("ctest/missing/model.gguf", NULL), JINFER_ENOTFOUND, LE, "is not cached");
    CHECK_NULL_ERROR(jinfer_model_load("ctest-nobody/no-such-GGUF:Q8_0", NULL), JINFER_ENOTFOUND, LE, "not cached");
    const char* garbage = TMP("garbage.gguf");
    write_file(garbage, "not a model at all", 18);
    CHECK_NULL_ERROR(jinfer_model_load(garbage, NULL), JINFER_EINVAL, LE, "");
    const char* json = TMP("tokenizer.json");
    write_file(json, "{\"model\": {}}", 13);
    CHECK_NULL_ERROR(jinfer_model_load(json, NULL), JINFER_EINVAL, LE, "is not a GGUF model file");
    const char* headless = TMP("headless.gguf");   /* a GGUF with no architecture: no model */
    write_file(headless, "GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 24);
    CHECK_NULL_ERROR(jinfer_model_load(headless, NULL), JINFER_EINVAL, LE, "general.architecture");
    const char* cut = TMP("cut.gguf");
    write_file(cut, "GGUF\x03\x00\x00\x00", 8);
    CHECK_NULL_ERROR(jinfer_model_load(cut, NULL), JINFER_EINVAL, LE, "");
    CHECK_NULL_ERROR(jinfer_model_load("example.com/owner/repo", NULL), JINFER_EINVAL, LE, "");
}

static void load_options(void) {
    offline();
    jinfer_model_options o;
    jinfer_model_options_init(&o);
    o.struct_size = sizeof o - 1;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "is too small: call jinfer_model_options_init first");
    jinfer_model_options_init(&o);
    o.companion_count = -1;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "options.companion_count -1 does not match options.companions");
    o.companion_count = 1;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "does not match options.companions");
    const char* nulls[] = {NULL};
    o.companions = nulls;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "companions[0] is NULL");
    static const char* const shapes[] = {"media", "=x", "media="};
    for (int i = 0; i < COUNT(shapes); i++) {
        const char* one[] = {shapes[i]};
        o.companions = one;
        CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "must read \"<capability>=<ref>\"");
    }
    const char* missing[] = {"media=/ctest/no/such/mmproj.gguf"};
    o.companions = missing;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_ENOTFOUND, LE, "no such file");
    /* a capability this architecture does not offer, on a file that exists */
    jinfer_model* m = load(LM);
    const char* info = jinfer_model_info(m);
    const char* at = strstr(info, "\"path\":\"");
    char companion[1024];
    snprintf(companion, sizeof companion, "bogus=%.*s", at ? (int) (strchr(at + 8, '"') - (at + 8)) : 0, at ? at + 8 : "");
    const char* unknown[] = {companion};
    o.companions = unknown;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "bogus");
    jinfer_model_free(m);
    /* context_capacity: 0 the default, JINFER_CONTEXT_MAXIMUM (-1) the maximum, below or past them refused */
    jinfer_model_options_init(&o);
    o.context_capacity = 1024;
    m = jinfer_model_load(LM, &o);
    CHECK(m != NULL);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_CONTEXT_CAPACITY), 1024);
    int32_t max = jinfer_model_property(m, JINFER_PROPERTY_MAX_CONTEXT_LENGTH);
    jinfer_model_free(m);
    o.context_capacity = JINFER_CONTEXT_MAXIMUM;   /* a state's memory comes with its first use: cheap to load */
    m = jinfer_model_load(LM, &o);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_CONTEXT_CAPACITY), max);
    jinfer_model_free(m);
    o.context_capacity = 0;
    m = jinfer_model_load(LM, &o);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_CONTEXT_CAPACITY), max < 4096 ? max : 4096);
    jinfer_model_free(m);
    o.context_capacity = -2;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "context_capacity must be 0 (the default), -1 (the model's maximum) or positive, got -2");
    o.context_capacity = max + 1;
    CHECK_NULL_ERROR(jinfer_model_load(LM, &o), JINFER_EINVAL, LE, "exceeds the model's");
    CHECK_LAST(JINFER_EINVAL, LE, "; -1 selects the maximum");   /* the C API's terms, not the Java API's */
}

/* ---- a language model ---- */

static void language_model(void) {
    offline();
    jinfer_model* m = load(LM);
    CHECK_CLEAR(LE);
    int32_t vocab = jinfer_model_property(m, JINFER_PROPERTY_VOCAB_SIZE);
    int32_t max = jinfer_model_property(m, JINFER_PROPERTY_MAX_CONTEXT_LENGTH);
    CHECK(vocab > 0 && max >= 4096);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_CONTEXT_CAPACITY), 4096);   /* the default, the maximum being larger */
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_EMBEDDING_DIMENSION), 0);   /* not applicable */
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_SAMPLE_RATE), 0);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_HAS_CHAT), 1);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_HAS_MEDIA), 0);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_HAS_EMBEDDING), 0);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_HAS_RERANK), 0);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_HAS_SPEECH), 0);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_HAS_TRANSCRIPTION), 0);
    static const int32_t unknown[] = {0, 6, 99, 106, -1, INT32_MAX};
    for (int i = 0; i < COUNT(unknown); i++) CHECK_ERROR(jinfer_model_property(m, unknown[i]), JINFER_EINVAL, LE, "unknown jinfer_model_property key");
    CHECK_ERROR(jinfer_model_property(NULL, JINFER_PROPERTY_VOCAB_SIZE), JINFER_EINVAL, LE, "model is NULL");
    const char* info = jinfer_model_info(m);
    CHECK(info[0] == '{' && info[strlen(info) - 1] == '}');
    CHECK_HAS(info, "\"kind\":\"language\"");
    CHECK_HAS(info, "\"capabilities\":[\"chat\"]");
    CHECK_STR(jinfer_model_info(NULL), "");               /* "" on error, the error kept */
    CHECK_LAST(JINFER_EINVAL, LE, "model is NULL");
    /* the tokenizer: the model's vocabulary, and it outlives the model */
    toknroll_tokenizer* t = jinfer_model_tokenizer(m);
    CHECK(t != NULL);
    CHECK_EQ(toknroll_vocabulary_size(t), vocab);
    CHECK_NULL_ERROR(jinfer_model_tokenizer(NULL), JINFER_EINVAL, LE, "model is NULL");
    jinfer_model_free(m);
    int32_t ids[16];
    CHECK(toknroll_encode(t, "Hello", 5, 0, ids, 16) > 0);
    toknroll_tokenizer_free(t);
    /* the other kinds' doors are closed, each with a reason */
    m = load(LM);
    const char* texts[] = {"a"};
    float out[4];
    CHECK_NULL_ERROR(jinfer_speak(m, "{\"input\":\"x\"}"), JINFER_EUNSUPPORTED, LE, "is a language model, not a speech model");
    CHECK_NULL_ERROR(jinfer_transcribe(m, out, 4, 16000), JINFER_EUNSUPPORTED, LE, "not a transcription model");
    CHECK_ERROR(jinfer_embed(m, texts, 1, 0, out, 4), JINFER_EUNSUPPORTED, LE, "not an embedding model");
    CHECK_ERROR(jinfer_rerank(m, "q", texts, 1, NULL, out, 1), JINFER_EUNSUPPORTED, LE, "not a reranking model");
    CHECK_ERROR(jinfer_rerank(m, "q", texts, 1, NULL, NULL, 0), JINFER_EUNSUPPORTED, LE, "not a reranking model");   /* sizing too */
    jinfer_model_free(m);
}

/* ---- the raw door: states ---- */

static void state(void) {
    offline();
    jinfer_model* m = load(LM);
    int32_t vocab = jinfer_model_property(m, JINFER_PROPERTY_VOCAB_SIZE);
    int32_t ids[64];
    int32_t n = ids_of(m, "The capital of France is", ids, 64);
    CHECK(n > 0);
    jinfer_state* s = jinfer_state_new(m, 0);
    CHECK(s != NULL);
    CHECK_EQ(jinfer_state_position(s), 0);
    CHECK_ERROR(jinfer_state_logits(s, NULL, 0), JINFER_EINVAL, LE, "nothing ingested yet");
    CHECK_EQ(jinfer_state_ingest(s, ids, n), n);
    CHECK_EQ(jinfer_state_position(s), n);
    CHECK_EQ(jinfer_state_ingest(s, ids, 0), n);         /* nothing more */
    CHECK_EQ(jinfer_state_ingest(s, NULL, 0), n);
    CHECK_EQ(jinfer_state_logits(s, NULL, 0), vocab);    /* sizes */
    float* logits = (float*) malloc((size_t) vocab * sizeof *logits);
    float* again = (float*) malloc((size_t) vocab * sizeof *again);
    CHECK_EQ(jinfer_state_logits(s, logits, vocab), vocab);
    int finite = 1;
    for (int i = 0; i < vocab; i++) finite &= logits[i] == logits[i] && logits[i] < 1e30f && logits[i] > -1e30f;
    CHECK(finite);
    for (int i = 0; i < 4; i++) again[i] = -7;
    CHECK_EQ(jinfer_state_logits(s, again, 3), vocab);   /* a prefix, the size returned */
    CHECK(again[0] == logits[0] && again[2] == logits[2] && again[3] == -7);
    /* a second state over the same tokens computes the same logits */
    jinfer_state* twin = jinfer_state_new(m, 0);
    CHECK_EQ(jinfer_state_ingest(twin, ids, n), n);
    CHECK_EQ(jinfer_state_logits(twin, again, vocab), vocab);
    CHECK(!memcmp(logits, again, (size_t) vocab * sizeof *logits));
    jinfer_state_free(twin);
    /* reset: back to nothing */
    CHECK_EQ(jinfer_state_reset(s), JINFER_OK);
    CHECK_EQ(jinfer_state_position(s), 0);
    CHECK_ERROR(jinfer_state_logits(s, logits, vocab), JINFER_EINVAL, LE, "nothing ingested yet");
    /* errors */
    int32_t outside[] = {1, vocab};
    CHECK_ERROR(jinfer_state_ingest(s, outside, 2), JINFER_EINVAL, LE, "outside [0,");
    int32_t negative[] = {-1};
    CHECK_ERROR(jinfer_state_ingest(s, negative, 1), JINFER_EINVAL, LE, "token id -1 outside");
    CHECK_EQ(jinfer_state_position(s), 0);               /* a refused ingest ingests nothing */
    CHECK_ERROR(jinfer_state_ingest(s, ids, -1), JINFER_EINVAL, LE, "count must not be negative");
    CHECK_ERROR(jinfer_state_ingest(s, NULL, 1), JINFER_EINVAL, LE, "ids is NULL");
    CHECK_EQ(jinfer_state_ingest(s, ids, 1), 1);
    CHECK_ERROR(jinfer_state_logits(s, logits, -1), JINFER_EINVAL, LE, "float_capacity must not be negative");
    CHECK_ERROR(jinfer_state_logits(s, NULL, 4), JINFER_EINVAL, LE, "out is NULL");
    CHECK_ERROR(jinfer_state_embedding(s, logits, vocab), JINFER_EUNSUPPORTED, LE, "is a language model, not an embedding model");
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_EMBEDDING_DIMENSION), 0);   /* the two agree */
    CHECK_ERROR(jinfer_state_ingest(NULL, ids, 1), JINFER_EINVAL, LE, "state is NULL");
    CHECK_ERROR(jinfer_state_position(NULL), JINFER_EINVAL, LE, "state is NULL");
    CHECK_ERROR(jinfer_state_reset(NULL), JINFER_EINVAL, LE, "state is NULL");
    CHECK_ERROR(jinfer_state_logits(NULL, logits, vocab), JINFER_EINVAL, LE, "state is NULL");
    CHECK_ERROR(jinfer_state_embedding(NULL, logits, vocab), JINFER_EINVAL, LE, "state is NULL");
    CHECK_NULL_ERROR(jinfer_state_new(NULL, 0), JINFER_EINVAL, LE, "model is NULL");
    /* a model passed where a state goes is no state */
    CHECK_ERROR(jinfer_state_position((jinfer_state*) m), JINFER_EINVAL, LE, "is not a state handle");
    jinfer_state_free(s);
    free(again);
    free(logits);
    jinfer_model_free(m);   /* the states are gone; the model with them */
}

static void state_capacity(void) {
    offline();
    jinfer_model* m = load(LM);
    int32_t max = jinfer_model_property(m, JINFER_PROPERTY_MAX_CONTEXT_LENGTH);
    jinfer_state* s = jinfer_state_new(m, 16);
    int32_t ids[17];
    int32_t n = ids_of(m, "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen", ids, 17);
    CHECK(n >= 17);
    CHECK_ERROR(jinfer_state_ingest(s, ids, 17), JINFER_EINVAL, LE, "ingesting 17 tokens at position 0 exceeds the capacity of 16");
    CHECK_EQ(jinfer_state_position(s), 0);
    CHECK_EQ(jinfer_state_ingest(s, ids, 10), 10);
    CHECK_ERROR(jinfer_state_ingest(s, ids, 7), JINFER_EINVAL, LE, "ingesting 7 tokens at position 10 exceeds the capacity of 16");
    CHECK_EQ(jinfer_state_ingest(s, ids, 6), 16);       /* exactly full */
    jinfer_state_free(s);
    CHECK_NULL_ERROR(jinfer_state_new(m, -2), JINFER_EINVAL, LE, "context_capacity must be 0 (the default), -1 (the model's maximum)");
    CHECK_NULL_ERROR(jinfer_state_new(m, max + 1), JINFER_EINVAL, LE, "exceeds the model's context length");
    jinfer_model_free(m);
    /* on a model loaded at 1024: 0 is those 1024, JINFER_CONTEXT_MAXIMUM the whole context */
    jinfer_model_options o;
    jinfer_model_options_init(&o);
    o.context_capacity = 1024;
    m = jinfer_model_load(LM, &o);
    static int32_t many[1025];
    for (int i = 0; i < 1025; i++) many[i] = ids[i % n];
    s = jinfer_state_new(m, 0);
    CHECK_ERROR(jinfer_state_ingest(s, many, 1025), JINFER_EINVAL, LE, "exceeds the capacity of 1024");
    jinfer_state_free(s);
    s = jinfer_state_new(m, JINFER_CONTEXT_MAXIMUM);   /* its memory comes as it fills */
    CHECK_EQ(jinfer_state_ingest(s, many, 1025), 1025);
    jinfer_state_free(s);
    jinfer_model_free(m);
}

/* two threads on one state: the second gets EBUSY, never a race */
typedef struct { jinfer_state* s; const int32_t* ids; int32_t n; volatile int stop; int wrong; } busy_box;
static void keep_ingesting(void* p) {
    busy_box* b = (busy_box*) p;
    while (!b->stop) {
        if (jinfer_state_reset(b->s) != JINFER_OK && jinfer_last_error(NULL)[0] == 0) b->wrong++;
        jinfer_state_ingest(b->s, b->ids, b->n);
    }
}

static void state_busy(void) {
    offline();
    jinfer_model* m = load(LM);
    static int32_t ids[512];
    char text[4096] = "";
    for (int i = 0; i < 200; i++) strcat(text, "word ");
    busy_box b = {jinfer_state_new(m, 0), ids, ids_of(m, text, ids, 512), 0, 0};
    thread_t worker = thread_run(keep_ingesting, &b);
    int busy = 0;
    char message[128] = "";
    for (int i = 0; i < 2000000 && !busy; i++) {
        if (jinfer_state_position(b.s) == JINFER_EBUSY) {
            busy = 1;
            snprintf(message, sizeof message, "%s", LE(NULL));
        }
    }
    b.stop = 1;
    thread_join(worker);
    CHECK(busy);
    CHECK_EQ(b.wrong, 0);
    CHECK_STR(message, "the state is in a call on another thread");
    jinfer_state_free(b.s);
    jinfer_model_free(m);
}

/* ---- streams ---- */

static void chat(void) {
    offline();
    jinfer_model* m = load(LM);
    jinfer_stream* s = jinfer_chat(m, HI);
    CHECK(s != NULL);
    CHECK_STR(jinfer_stream_result(s), "");            /* not finished */
    CHECK_LAST(JINFER_EINVAL, LE, "the stream has not finished");
    char text[512];
    size_t at = 0;
    int32_t rc, chunks = 0;
    while ((rc = jinfer_stream_step(s)) == JINFER_OK) {
        int32_t count = 99;
        CHECK(jinfer_stream_samples(s, &count) == NULL && count == 0);   /* text, no audio */
        append(text, sizeof text, &at, jinfer_stream_text(s));
        chunks++;
    }
    CHECK_EQ(rc, JINFER_DONE);
    CHECK(chunks > 0);
    CHECK_HAS(text, "Paris");
    CHECK_EQ(jinfer_stream_step(s), JINFER_DONE);       /* and stays done */
    const char* result = jinfer_stream_result(s);
    CHECK_HAS(result, "\"object\":\"chat.completion\"");
    CHECK_HAS(result, "\"finish_reason\":");
    CHECK_HAS(result, "\"prompt_tokens\":");
    CHECK_HAS(result, "Paris");
    jinfer_stream_free(s);
    /* the same request, the same reply */
    s = jinfer_chat(m, HI);
    char twice[512];
    CHECK_EQ(drain(s, twice, sizeof twice), JINFER_DONE);
    CHECK_STR(twice, text);
    jinfer_stream_free(s);
    /* max_tokens 1: the length ran out */
    s = jinfer_chat(m, "{\"messages\":[{\"role\":\"user\",\"content\":\"Count to ten.\"}],\"max_tokens\":1}");
    CHECK_EQ(drain(s, twice, sizeof twice), JINFER_DONE);
    CHECK_HAS(jinfer_stream_result(s), "\"finish_reason\":\"length\"");
    CHECK_HAS(jinfer_stream_result(s), "\"completion_tokens\":1");
    jinfer_stream_free(s);
    jinfer_model_free(m);   /* a stream keeps its model alive: freed after it here */
}

static void chat_errors(void) {
    offline();
    jinfer_model* m = load(LM);
    CHECK_NULL_ERROR(jinfer_chat(NULL, HI), JINFER_EINVAL, LE, "model is NULL");
    CHECK_NULL_ERROR(jinfer_chat(m, NULL), JINFER_EINVAL, LE, "request_json is NULL");
    CHECK_NULL_ERROR(jinfer_chat(m, ""), JINFER_EINVAL, LE, "request_json is not valid JSON");
    CHECK_NULL_ERROR(jinfer_chat(m, "nope"), JINFER_EINVAL, LE, "request_json is not valid JSON");
    char* wide = (char*) malloc(200001);   /* its parser echoes the line: the message stays one line, 1 KB */
    memset(wide, 'x', 200000);
    wide[0] = '{';
    wide[200000] = 0;
    CHECK_NULL_ERROR(jinfer_chat(m, wide), JINFER_EINVAL, LE, "request_json is not valid JSON");
    CHECK(strlen(LE(NULL)) <= 1024 && !strchr(LE(NULL), '\n'));
    free(wide);
    CHECK_NULL_ERROR(jinfer_chat(m, "[1]"), JINFER_EINVAL, LE, "request_json must be a JSON object");
    CHECK_NULL_ERROR(jinfer_chat(m, "{}"), JINFER_EINVAL, LE, "messages must be an array");
    CHECK_NULL_ERROR(jinfer_chat(m, "{\"messages\":\"hi\"}"), JINFER_EINVAL, LE, "messages must be an array");
    CHECK_NULL_ERROR(jinfer_chat(m, "{\"messages\":[]}"), JINFER_EINVAL, LE, "messages must not be empty");
    CHECK_ERROR(jinfer_prepare(m, "nope"), JINFER_EINVAL, LE, "request_json is not valid JSON");
    CHECK_ERROR(jinfer_stream_step(NULL), JINFER_EINVAL, LE, "stream is NULL");
    CHECK_STR(jinfer_stream_text(NULL), "");
    CHECK_STR(jinfer_stream_result(NULL), "");
    int32_t count = 99;
    CHECK(jinfer_stream_samples(NULL, &count) == NULL && count == 0);
    CHECK(jinfer_stream_samples(NULL, NULL) == NULL);
    jinfer_model_free(m);
    /* a prompt past the context capacity */
    jinfer_model_options o;
    jinfer_model_options_init(&o);
    o.context_capacity = 1024;
    m = jinfer_model_load(LM, &o);
    static char big[16384];
    int at = snprintf(big, sizeof big, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    for (int i = 0; i < 1200; i++) at += snprintf(big + at, sizeof big - (size_t) at, "word ");
    snprintf(big + at, sizeof big - (size_t) at, "\"}],\"max_tokens\":4}");
    jinfer_stream* s = jinfer_chat(m, big);
    int32_t rc = s ? jinfer_stream_step(s) : JINFER_EINVAL;
    CHECK_ERROR(rc, JINFER_EINVAL, LE, "1024");
    jinfer_stream_free(s);
    jinfer_model_free(m);
}

static void cancel_on_another_thread(void* s) { jinfer_stream_cancel((jinfer_stream*) s); }

static void cancel(void) {
    offline();
    jinfer_model* m = load(LM);
    char text[8192];
    /* before the first step */
    jinfer_stream* s = jinfer_chat(m, STORY);
    jinfer_stream_cancel(s);
    CHECK_ERROR(jinfer_stream_step(s), JINFER_ECANCELLED, LE, "the stream was cancelled");
    CHECK_HAS(jinfer_stream_result(s), "\"finish_reason\":\"cancelled\"");
    jinfer_stream_cancel(s);                           /* twice is harmless */
    jinfer_stream_free(s);
    /* from another thread, mid-reply: the next step stops, and the text so far is the result's */
    s = jinfer_chat(m, STORY);
    size_t at = 0;
    text[0] = 0;
    for (int i = 0; i < 4; i++) {
        CHECK_EQ(jinfer_stream_step(s), JINFER_OK);
        append(text, sizeof text, &at, jinfer_stream_text(s));
    }
    thread_join(thread_run(cancel_on_another_thread, s));
    CHECK_ERROR(jinfer_stream_step(s), JINFER_ECANCELLED, LE, "the stream was cancelled");
    const char* result = jinfer_stream_result(s);
    CHECK_HAS(result, "\"finish_reason\":\"cancelled\"");
    CHECK(at > 0);
    CHECK_HAS(result, text);
    jinfer_stream_free(s);
    /* after it finished: nothing changes */
    s = jinfer_chat(m, HI);
    CHECK_EQ(drain(s, text, sizeof text), JINFER_DONE);
    jinfer_stream_cancel(s);
    CHECK_EQ(jinfer_stream_step(s), JINFER_DONE);
    CHECK_HAS(jinfer_stream_result(s), "Paris");
    jinfer_stream_free(s);
    /* freed while generating: cancelled, no wait for the rest */
    s = jinfer_chat(m, STORY);
    CHECK_EQ(jinfer_stream_step(s), JINFER_OK);
    jinfer_stream_free(s);
    /* the model serves on after all of it */
    s = jinfer_chat(m, HI);
    CHECK_EQ(drain(s, text, sizeof text), JINFER_DONE);
    CHECK_HAS(text, "Paris");
    jinfer_stream_free(s);
    jinfer_model_free(m);
}

/* two threads stepping one stream: the second gets EBUSY, never a race */
typedef struct { jinfer_stream* s; volatile int stepping; int32_t rc; } step_box;
static void keep_stepping(void* p) {
    step_box* b = (step_box*) p;
    b->stepping = 1;
    while ((b->rc = jinfer_stream_step(b->s)) == JINFER_OK || b->rc == JINFER_EBUSY) {}
}

static void stream_busy(void) {
    offline();
    jinfer_model* m = load(LM);
    static char big[8192];
    int at = snprintf(big, sizeof big, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    for (int i = 0; i < 600; i++) at += snprintf(big + at, sizeof big - (size_t) at, "word ");   /* a long prefill */
    snprintf(big + at, sizeof big - (size_t) at, "\"}],\"max_tokens\":32,\"temperature\":0}");
    step_box b = {jinfer_chat(m, big), 0, 0};
    thread_t worker = thread_run(keep_stepping, &b);
    while (!b.stepping) {}
    int busy = 0;
    char message[128] = "";
    for (int i = 0; i < 20000000 && !busy; i++) {
        int32_t rc = jinfer_stream_step(b.s);
        if (rc == JINFER_EBUSY) {
            busy = 1;
            snprintf(message, sizeof message, "%s", LE(NULL));
        } else if (rc != JINFER_OK) break;
    }
    thread_join(worker);
    CHECK(busy);
    CHECK_STR(message, "the stream is in a call on another thread");
    CHECK_EQ(b.rc, JINFER_DONE);
    jinfer_stream_free(b.s);
    jinfer_model_free(m);
}

static void prepare(void) {
    offline();
    jinfer_model* m = load(LM);
    static const char* system = "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a terse geography tutor who answers in one word, always, without exception.\"}]}";
    CHECK_EQ(jinfer_prepare(m, system), JINFER_OK);
    CHECK_CLEAR(LE);
    jinfer_stream* s = jinfer_chat(m, "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a terse geography tutor who answers in one word, always, without exception.\"},"
                                      "{\"role\":\"user\",\"content\":\"Capital of Italy?\"}],\"temperature\":0,\"max_tokens\":4}");
    char text[256];
    CHECK_EQ(drain(s, text, sizeof text), JINFER_DONE);
    const char* cached = strstr(jinfer_stream_result(s), "\"cached_tokens\":");
    CHECK(cached && atoi(cached + 16) > 0);              /* the prefix was served from the cache */
    jinfer_stream_free(s);
    CHECK_ERROR(jinfer_prepare(m, NULL), JINFER_EINVAL, LE, "request_json is NULL");
    CHECK_ERROR(jinfer_prepare(m, "{}"), JINFER_EINVAL, LE, "messages must be an array");
    CHECK_ERROR(jinfer_prepare(NULL, system), JINFER_EINVAL, LE, "model is NULL");
    jinfer_model_free(m);
}

/* a reply and a stream on two threads at once: one model serves both */
typedef struct { jinfer_model* m; int32_t rc; char text[256]; } chat_box;
static void chat_worker(void* p) {
    chat_box* c = (chat_box*) p;
    jinfer_stream* s = jinfer_chat(c->m, HI);
    c->rc = s ? drain(s, c->text, sizeof c->text) : JINFER_ERROR;
    jinfer_stream_free(s);
}
typedef struct { jinfer_model* m; const int32_t* ids; int32_t n; float first; int wrong; } logits_box;
static void logits_worker(void* p) {
    logits_box* b = (logits_box*) p;
    int32_t vocab = jinfer_model_property(b->m, JINFER_PROPERTY_VOCAB_SIZE);
    float* out = (float*) malloc((size_t) vocab * sizeof *out);
    for (int i = 0; i < 3; i++) {
        jinfer_state* s = jinfer_state_new(b->m, 64);
        if (jinfer_state_ingest(s, b->ids, b->n) != b->n || jinfer_state_logits(s, out, vocab) != vocab) b->wrong++;
        if (i == 0) b->first = out[0]; else if (out[0] != b->first) b->wrong++;
        jinfer_state_free(s);
    }
    free(out);
}

static void threads(void) {
    offline();
    jinfer_model* m = load(LM);
    int32_t ids[32];
    int32_t n = ids_of(m, "The capital of France is", ids, 32);
    chat_box chats[2] = {{m, 0, ""}, {m, 0, ""}};
    logits_box states[3] = {{m, ids, n, 0, 0}, {m, ids, n, 0, 0}, {m, ids, n, 0, 0}};
    thread_t running[5];
    for (int i = 0; i < 2; i++) running[i] = thread_run(chat_worker, &chats[i]);
    for (int i = 0; i < 3; i++) running[2 + i] = thread_run(logits_worker, &states[i]);
    for (int i = 0; i < 5; i++) thread_join(running[i]);
    for (int i = 0; i < 2; i++) {
        CHECK_EQ(chats[i].rc, JINFER_DONE);
        CHECK_HAS(chats[i].text, "Paris");
    }
    for (int i = 0; i < 3; i++) CHECK_EQ(states[i].wrong, 0);
    CHECK(states[0].first == states[1].first && states[1].first == states[2].first);
    jinfer_model_free(m);
}

/* ---- an embedding model ---- */

static void embedding_model(void) {
    offline();
    jinfer_model* e = load(EMB);
    int32_t dim = jinfer_model_property(e, JINFER_PROPERTY_EMBEDDING_DIMENSION);
    CHECK(dim > 0);
    CHECK_EQ(jinfer_model_property(e, JINFER_PROPERTY_HAS_EMBEDDING), 1);
    CHECK_EQ(jinfer_model_property(e, JINFER_PROPERTY_HAS_CHAT), 0);
    int32_t max = jinfer_model_property(e, JINFER_PROPERTY_MAX_CONTEXT_LENGTH);
    CHECK_EQ(jinfer_model_property(e, JINFER_PROPERTY_CONTEXT_CAPACITY), max < 4096 ? max : 4096);   /* the default, an embedder's too */
    CHECK_HAS(jinfer_model_info(e), "\"kind\":\"embedding\"");
    const char* texts[] = {"a cat sat on the mat", "the stock market fell", "a cat sat on the mat"};
    CHECK_EQ(jinfer_embed(e, texts, 3, 0, NULL, 0), 3 * dim);   /* sizes */
    float* out = (float*) malloc((size_t) (3 * dim + 1) * sizeof *out);
    out[3 * dim] = -7;
    CHECK_EQ(jinfer_embed(e, texts, 3, 0, out, 3 * dim), 3 * dim);
    CHECK_EQ(out[3 * dim], -7);
    CHECK(!memcmp(out, out + 2 * dim, (size_t) dim * sizeof *out));   /* the same text, the same vector */
    CHECK(memcmp(out, out + dim, (size_t) dim * sizeof *out));
    float first = out[0];
    out[1] = -7;
    CHECK_EQ(jinfer_embed(e, texts, 3, 0, out, 1), 3 * dim);   /* a prefix */
    CHECK(out[0] == first && out[1] == -7);
    CHECK_EQ(jinfer_embed(e, texts, 3, dim, out, 3 * dim), 3 * dim);   /* the native dimension, asked by name */
    CHECK_EQ(jinfer_embed(e, NULL, 0, 0, NULL, 0), 0);
    /* errors */
    CHECK_ERROR(jinfer_embed(NULL, texts, 1, 0, out, dim), JINFER_EINVAL, LE, "model is NULL");
    CHECK_ERROR(jinfer_embed(e, NULL, 1, 0, out, dim), JINFER_EINVAL, LE, "texts is NULL");
    const char* holes[] = {"a", NULL};
    CHECK_ERROR(jinfer_embed(e, holes, 2, 0, out, 2 * dim), JINFER_EINVAL, LE, "texts[1] is NULL");
    CHECK_ERROR(jinfer_embed(e, texts, -1, 0, out, dim), JINFER_EINVAL, LE, "count must not be negative");
    CHECK_ERROR(jinfer_embed(e, texts, 1, -1, out, dim), JINFER_EINVAL, LE, "dimension must not be negative; 0 = native");
    CHECK_ERROR(jinfer_embed(e, texts, 1, dim + 1, out, dim), JINFER_EINVAL, LE, "requested");
    CHECK_ERROR(jinfer_embed(e, texts, 1, 0, out, -1), JINFER_EINVAL, LE, "float_capacity must not be negative");
    CHECK_ERROR(jinfer_embed(e, texts, 1, 0, NULL, dim), JINFER_EINVAL, LE, "out is NULL");
    /* its state: a pooled vector, and logits too; one packed batch at most */
    int32_t ids[8];
    int32_t n = ids_of(e, "a cat", ids, 8);
    jinfer_state* s = jinfer_state_new(e, 0);
    CHECK_ERROR(jinfer_state_embedding(s, out, dim), JINFER_EINVAL, LE, "nothing ingested yet");
    CHECK_EQ(jinfer_state_ingest(s, ids, n), n);
    CHECK_EQ(jinfer_state_embedding(s, NULL, 0), dim);
    CHECK_EQ(jinfer_state_embedding(s, out, dim), dim);
    static int32_t many[600];
    for (int i = 0; i < 600; i++) many[i] = ids[0];
    CHECK_EQ(jinfer_state_reset(s), JINFER_OK);
    CHECK_ERROR(jinfer_state_ingest(s, many, 600), JINFER_EINVAL, LE, "an embedding model ingests at most");
    jinfer_state_free(s);
    /* the doors it has not */
    CHECK_NULL_ERROR(jinfer_chat(e, HI), JINFER_EUNSUPPORTED, LE, "is an embedding model, not a language model");
    CHECK_ERROR(jinfer_prepare(e, HI), JINFER_EUNSUPPORTED, LE, "not a language model");
    CHECK_NULL_ERROR(jinfer_speak(e, "{\"input\":\"x\"}"), JINFER_EUNSUPPORTED, LE, "not a speech model");
    CHECK_ERROR(jinfer_rerank(e, "q", texts, 1, NULL, out, 1), JINFER_EUNSUPPORTED, LE, "not a reranking model");
    toknroll_tokenizer* t = jinfer_model_tokenizer(e);
    CHECK(t != NULL);
    toknroll_tokenizer_free(t);
    free(out);
    jinfer_model_free(e);
}

/* ---- the optional kinds ---- */

static void reranker(void) {
    offline();
    jinfer_model* r = cached_model(RERANKER, NULL);
    CHECK_EQ(jinfer_model_property(r, JINFER_PROPERTY_HAS_RERANK), 1);
    int32_t max = jinfer_model_property(r, JINFER_PROPERTY_MAX_CONTEXT_LENGTH);
    CHECK_EQ(jinfer_model_property(r, JINFER_PROPERTY_CONTEXT_CAPACITY), max < 4096 ? max : 4096);   /* the default, a reranker's too */
    CHECK_EQ(jinfer_model_property(r, JINFER_PROPERTY_HAS_CHAT), 0);
    const char* docs[] = {"Paris is the capital of France.", "Bananas are yellow."};
    float scores[3] = {0, 0, -7};
    CHECK_EQ(jinfer_rerank(r, "capital of France", docs, 2, NULL, scores, 3), 2);
    CHECK(scores[0] > scores[1] && scores[2] == -7);
    CHECK_EQ(jinfer_rerank(r, "capital of France", docs, 2, "Judge whether the document answers the query.", scores, 2), 2);
    CHECK(scores[0] > scores[1]);
    /* capacity 0 sizes, before any work; too little room writes nothing */
    CHECK_EQ(jinfer_rerank(r, "capital of France", docs, 2, NULL, NULL, 0), 2);
    scores[0] = -7;
    CHECK_EQ(jinfer_rerank(r, "capital of France", docs, 2, NULL, scores, 1), 2);
    CHECK(scores[0] == -7);
    CHECK_EQ(jinfer_rerank(r, "q", NULL, 0, NULL, NULL, 0), 0);
    CHECK_ERROR(jinfer_rerank(r, NULL, docs, 2, NULL, scores, 2), JINFER_EINVAL, LE, "query is NULL");
    CHECK_ERROR(jinfer_rerank(r, "q", NULL, 2, NULL, scores, 2), JINFER_EINVAL, LE, "documents is NULL");
    CHECK_ERROR(jinfer_rerank(r, "q", docs, 2, NULL, NULL, 2), JINFER_EINVAL, LE, "out is NULL");
    CHECK_ERROR(jinfer_rerank(r, "q", docs, 2, NULL, scores, -1), JINFER_EINVAL, LE, "float_capacity must not be negative");
    CHECK_ERROR(jinfer_rerank(r, "q", docs, -1, NULL, scores, 2), JINFER_EINVAL, LE, "count must not be negative");
    CHECK_NULL_ERROR(jinfer_model_tokenizer(r), JINFER_EUNSUPPORTED, LE, "it exposes no tokenizer");
    CHECK_NULL_ERROR(jinfer_chat(r, HI), JINFER_EUNSUPPORTED, LE, "not a language model");
    jinfer_model_free(r);
}

static void speech_and_transcription(void) {
    offline();
    jinfer_model* tts = cached_model(SPEAKER, VOICE);
    jinfer_model* asr = cached_model(TRANSCRIBER, NULL);
    static float audio[24000 * 20];
    int32_t samples = 0, rate = jinfer_model_property(tts, JINFER_PROPERTY_SAMPLE_RATE);
    CHECK_EQ(rate, 24000);
    CHECK_EQ(jinfer_model_property(tts, JINFER_PROPERTY_HAS_SPEECH), 1);
    jinfer_stream* s = jinfer_speak(tts, "{\"input\":\"Turns out, matrix multiplications can talk.\",\"speed\":1.2}");
    CHECK(s != NULL);
    int32_t rc, chunks = 0;
    while ((rc = jinfer_stream_step(s)) == JINFER_OK) {
        int32_t count = 0;
        const float* pcm = jinfer_stream_samples(s, &count);
        CHECK(pcm != NULL && count > 0);
        if (pcm && samples + count <= (int32_t) (sizeof audio / sizeof *audio)) memcpy(audio + samples, pcm, (size_t) count * sizeof *pcm);
        samples += count;
        chunks++;
    }
    CHECK_EQ(rc, JINFER_DONE);
    CHECK(chunks > 0 && samples > rate && samples < 10 * rate);   /* a second to ten */
    CHECK(jinfer_stream_result(s)[0] == '{');
    jinfer_stream_free(s);
    CHECK_NULL_ERROR(jinfer_speak(tts, "{}"), JINFER_EINVAL, LE, "request_json needs a non-empty \"input\" string");
    CHECK_NULL_ERROR(jinfer_speak(tts, "{\"input\":\"\"}"), JINFER_EINVAL, LE, "non-empty \"input\"");
    CHECK_NULL_ERROR(jinfer_speak(tts, "{\"input\":\"x\",\"speed\":\"fast\"}"), JINFER_EINVAL, LE, "\"speed\" must be a number");
    CHECK_NULL_ERROR(jinfer_speak(tts, "{\"input\":\"x\",\"speed\":0}"), JINFER_EINVAL, LE, "\"speed\" must be positive");
    CHECK_NULL_ERROR(jinfer_speak(tts, NULL), JINFER_EINVAL, LE, "request_json is NULL");
    CHECK_NULL_ERROR(jinfer_speak(NULL, "{\"input\":\"x\"}"), JINFER_EINVAL, LE, "model is NULL");
    CHECK_NULL_ERROR(jinfer_chat(tts, HI), JINFER_EUNSUPPORTED, LE, "not a language model");
    /* cancelled mid-speech */
    s = jinfer_speak(tts, "{\"input\":\"A much longer sentence, spoken slowly, so that there is time to stop it in the middle of a word.\"}");
    jinfer_stream_cancel(s);
    CHECK_EQ(jinfer_stream_step(s), JINFER_ECANCELLED);
    jinfer_stream_free(s);
    CHECK_EQ(jinfer_model_property(asr, JINFER_PROPERTY_SAMPLE_RATE), 16000);
    CHECK_EQ(jinfer_model_property(asr, JINFER_PROPERTY_HAS_TRANSCRIPTION), 1);
    float silence[1600] = {0};
    CHECK_NULL_ERROR(jinfer_transcribe(asr, silence, 1600, 0), JINFER_EINVAL, LE, "sample_rate must be positive");
    CHECK_NULL_ERROR(jinfer_transcribe(asr, silence, 1600, -16000), JINFER_EINVAL, LE, "sample_rate must be positive");
    CHECK_NULL_ERROR(jinfer_transcribe(asr, NULL, 10, 16000), JINFER_EINVAL, LE, "pcm is NULL");
    CHECK_NULL_ERROR(jinfer_transcribe(asr, silence, -1, 16000), JINFER_EINVAL, LE, "count must not be negative");
    CHECK_NULL_ERROR(jinfer_transcribe(NULL, silence, 1600, 16000), JINFER_EINVAL, LE, "model is NULL");
    /* what the speaker said, at its own rate: the library resamples */
    s = jinfer_transcribe(asr, audio, samples, rate);
    char text[1024];
    CHECK_EQ(drain(s, text, sizeof text), JINFER_DONE);
    const char* result = jinfer_stream_result(s);
    CHECK(strstr(result, "matrix") || strstr(result, "Matrix"));
    jinfer_stream_free(s);
    jinfer_model_free(tts);
    jinfer_model_free(asr);
}

/* ---- fork ---- */

#ifndef _WIN32
static void fork_refused(void) {
    offline();
    jinfer_model* m = load(LM);
    toknroll_tokenizer* t = jinfer_model_tokenizer(m);
    alarm(120);
    pid_t pid = fork();
    if (pid == 0) {
        int bad = 0;
        bad += jinfer_model_property(m, JINFER_PROPERTY_VOCAB_SIZE) != JINFER_ERROR;
        bad += !strstr(LE(NULL), "fork");
        bad += jinfer_chat(m, HI) != NULL;
        bad += jinfer_model_load(LM, NULL) != NULL;
        int32_t ids[8];
        bad += toknroll_encode(t, "x", 1, 0, ids, 8) != TOKNROLL_ERROR;
        jinfer_model_free(m);         /* does nothing */
        toknroll_tokenizer_free(t);
        bad += !*jinfer_version();
        _exit(bad);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK_EQ(jinfer_model_property(m, JINFER_PROPERTY_HAS_CHAT), 1);   /* the parent serves on */
    toknroll_tokenizer_free(t);
    jinfer_model_free(m);
}
#endif

/* a heap too small for the model: out of memory names the knob, recorded without allocating on the full heap,
 * and the library goes on */
static void heap_too_small(void) {
    const char* options[] = {"-Djinfer.offline=true", "-XX:MaxHeapSize=16m"};   /* configure is once: both at once */
    CHECK_EQ(jinfer_configure(options, 2), JINFER_OK);
    jinfer_model* m = jinfer_model_load(LM, NULL);
    if (m) { jinfer_model_free(m); FAIL_AT("%s loaded in a 16 MB heap: lower the test's heap", LM); return; }
    int32_t code = 0;
    const char* message = LE(&code);
    if (code == JINFER_ENOTFOUND) skip("%s is not cached (%s)", LM, message);
    CHECK_EQ(code, JINFER_ERROR);
    CHECK_HAS(message, "out of memory");
    CHECK_HAS(message, "-XX:MaxHeapSize");
    CHECK(jinfer_model_load("/no/such/model.gguf", NULL) == NULL);   /* the library still answers, in words */
    LE(&code);
    CHECK_EQ(code, JINFER_ENOTFOUND);
}

static const scenario scenarios[] = {
    {"version", version, NULL},
    {"configure", configure, NULL},
    {"configure_through_toknroll", configure_through_toknroll, NULL},
    {"last_error_before_start", last_error_before_start, NULL},
    {"error_channels", error_channels, NULL},
    {"configure_rejected", configure_rejected, NULL},
    {"heap_too_small", heap_too_small, NULL},
    {"options_from_the_environment", options_from_the_environment, "JINFER_OPTIONS=-Djinfer.threads=3"},
    {"log_set_once", log_set_once, NULL},
    {"log_set_through_toknroll", log_set_through_toknroll, NULL},
    {"log_set_after_a_call", log_set_after_a_call, NULL},
    {"log_quiet_by_default", log_quiet_by_default, NULL},
    {"log_quiet_on_stderr", log_quiet_on_stderr, NULL},
    {"log_callback_calls_in_at_start", log_callback_calls_in_at_start, "JINFER_LOG=loud"},
    {"log_debug_to_the_callback", log_debug_to_the_callback, "JINFER_LOG=debug"},
    {"log_debug_to_stderr", log_debug_to_stderr, "JINFER_LOG=DEBUG"},
    {"log_null_is_silence", log_null_is_silence, "JINFER_LOG=debug"},
    {"log_off", log_off, "JINFER_LOG=off"},
    {"log_unknown_value", log_unknown_value, "JINFER_LOG=loud"},
    {"log_toknroll_log_is_not_read", log_toknroll_log_is_not_read, "TOKNROLL_LOG=bogus"},
    {"log_callback_may_free", log_callback_may_free, "JINFER_LOG=debug"},
    {"load_errors", load_errors, NULL},
    {"load_options", load_options, NULL},
    {"language_model", language_model, NULL},
    {"state", state, NULL},
    {"state_capacity", state_capacity, NULL},
    {"state_busy", state_busy, NULL},
    {"chat", chat, NULL},
    {"chat_errors", chat_errors, NULL},
    {"cancel", cancel, NULL},
    {"stream_busy", stream_busy, NULL},
    {"prepare", prepare, NULL},
    {"threads", threads, NULL},
    {"embedding_model", embedding_model, NULL},
    {"reranker", reranker, NULL},
    {"speech_and_transcription", speech_and_transcription, NULL},
#ifndef _WIN32
    {"fork_refused", fork_refused, NULL},
#endif
};

int main(int argc, char** argv) {
    return test_main(argc, argv, scenarios, COUNT(scenarios));
}
