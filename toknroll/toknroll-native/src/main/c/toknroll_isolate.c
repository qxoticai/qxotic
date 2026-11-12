/* The isolate behind the toknroll__* entry points: created by configure, or else on first use, with the
 * options configure passed plus those in the environment, each calling thread attached once and detached
 * when it exits. libjinfer compiles this file too (TOKNROLL_PROGRAM "jinfer", TOKNROLL_OPTIONS_ENV
 * "JINFER_OPTIONS", TOKNROLL_SELF_PROPERTY "jam.native.library.path", TOKNROLL_FORK_REFUSED), so its
 * jinfer_* and toknroll_* symbols share one isolate and one configure, and libjam, linked into it, is
 * loaded from the image.
 *
 * Fork. A child inherits the runtime's locks as they were at fork, and a lock some other thread held has no
 * owner left in the child to release it: the child hangs on its first call. libtoknroll closes the window:
 * a thread marks itself busy for the length of each call, and fork waits until no thread is, so no call is
 * inside the runtime when the process forks and the child carries on with every tokenizer it inherited.
 * Each mark sits in its own thread's record, so a call writes no memory another thread reads (a shared
 * read lock cost 18x per call at 8 threads). ponytail: the runtime's own Reference Handler thread is not
 * gated; 200 forks never caught it holding a lock. libjinfer (TOKNROLL_FORK_REFUSED) generates on threads
 * of its own, which a child does not inherit, and calls back into the host while holding a call open, so
 * it takes no gate: a forked child is refused, every call failing with the reason, every free a no-op,
 * where it would hang. Windows has no fork.
 *
 * Batch workers. A thread of the runtime caught running Java code at fork hangs the child at its first
 * garbage collection, which waits for every thread the runtime knows, gone or not. So libtoknroll's batches
 * run on workers of its own, held to the same gate: a worker runs Java code only while it runs a task, marked
 * busy, and waits for work here, in C, where fork finds it idle. The child has none of them, and encodes its
 * batches on the calling thread. The workers serve every build, libjinfer's and Windows' too (below). */
#define _GNU_SOURCE               /* dladdr */
#define _DARWIN_C_SOURCE
#ifndef _WIN32
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toknroll.h"
#include "toknroll_isolate.h"
#include "toknroll_port.h"

_Static_assert(sizeof(void*) == 8 && sizeof(size_t) == 8, "the C API assumes a 64-bit platform");

/* Bare names, stringified here: a quoted -D value would not survive every platform's command line. */
#ifndef TOKNROLL_PROGRAM
#define TOKNROLL_PROGRAM toknroll                /* argv[0] */
#endif
#ifndef TOKNROLL_OPTIONS_ENV
#define TOKNROLL_OPTIONS_ENV TOKNROLL_OPTIONS    /* options separated by whitespace, applied after configure's */
#endif
/* TOKNROLL_SELF_PROPERTY, when defined (libjinfer: jam.native.library.path): the runtime is told this
 * image's own file as that property, so a native library linked INTO the image (libjam) is loaded from
 * the image itself, through the ordinary System.load path every platform resolves symbols on. */

/* graal_isolate.h, spelled out because it exists only once the image is built, after this file is
 * compiled. graal_isolate is graal_isolate_t, opaque. isolate_params is graal_create_isolate_params_t,
 * version 5, as the generated header documents it: argv is "parsed like command line arguments" (argv[0]
 * is the program), and unrecognized arguments fail creation unless ignored (they are not: a typo must
 * fail, not do nothing). */
typedef struct graal_isolate graal_isolate;
typedef struct {
    int version;
    uintptr_t reserved_address_space_size;
    const char* auxiliary_image_path;             /* internal */
    uintptr_t auxiliary_image_reserved_space_size; /* internal */
    int argc;
    char** argv;
    int pkey;                                     /* internal */
    char ignore_unrecognized_args;
    char reserved_4;
    char reserved_5;
} isolate_params;
int graal_create_isolate(isolate_params* params, graal_isolate** isolate, graal_thread** thread);
int graal_attach_thread(graal_isolate* isolate, graal_thread** thread);
int graal_detach_thread(graal_thread* thread);
graal_thread* graal_get_current_thread(graal_isolate* isolate);

static graal_isolate* isolate;
static port_once_t once = PORT_ONCE_INIT;
static port_tls_t key;
static char program[] = STR(TOKNROLL_PROGRAM);   /* argv[0]; GraalVM parses the rest like a java command line */
static char** options;                  /* configure's, copied */
static int option_count;
static const char* configured_by;       /* the public function configure came through, once */
static long created;                    /* set when creation starts, port_load/port_store: then configure is too late */
static long started;                    /* the runtime is up and this file's state with it: port_load/port_store */
/* a failure with no isolate to hold the message, before it or in a forked child: the calling thread's own */
static PORT_THREAD_LOCAL char early_error[1024];   /* the Java side's bound (Errors.LIMIT) */
static PORT_THREAD_LOCAL int32_t early_code;
static char start_error[1024];          /* why there is no runtime, for every later call to say again */
static int32_t start_code;

/* The runtime APIs' error text: allocated only on a thread's first error, reused, freed at thread exit.
 * Independent of isolate startup: the PGO training program calls the Java entry points directly too. */
static port_once_t error_once = PORT_ONCE_INIT;
static port_tls_t error_keys[2];   /* toknroll, jinfer */
static int error_keys_ready;

static void error_keys_init(void) {
    if (port_tls_create(&error_keys[0], free) != 0) return;
    if (port_tls_create(&error_keys[1], free) != 0) { port_tls_delete(error_keys[0]); return; }
    error_keys_ready = 1;
}

static const char* error_buffer(int api, int32_t allocate) {
    static const char unavailable[] = "Error message unavailable: out of memory";
    port_once(&error_once, error_keys_init);
    char* buffer = error_keys_ready ? port_tls_get(error_keys[api]) : NULL;
    if (!buffer && allocate && error_keys_ready) {
        buffer = malloc(1025);   /* Errors.LIMIT UTF-8 bytes and the terminating NUL */
        if (buffer && port_tls_set(error_keys[api], buffer) != 0) { free(buffer); buffer = NULL; }
    }
    return buffer ? buffer : allocate ? NULL : unavailable;
}

const char* toknroll__error_buffer(int32_t allocate) { return error_buffer(0, allocate); }
const char* jinfer__error_buffer(int32_t allocate) { return error_buffer(1, allocate); }

/* The calling thread's last toknroll error code, beside its message: here and not in a Java thread-local,
 * whose first use allocates, so a call that fails out of memory can still say so. */
static PORT_THREAD_LOCAL int32_t error_code;
int32_t toknroll__error_code(void) { return error_code; }
void toknroll__error_code_set(int32_t code) { error_code = code; }

/* Set by toknroll__escaped, read and cleared by toknroll__took_escape: the C shim quarantines the call's
 * tokenizer when its call ended so (toknroll_shim.c). */
static PORT_THREAD_LOCAL int escape_seen;
int toknroll__took_escape(void) { int seen = escape_seen; escape_seen = 0; return seen; }

/* The words for an error that escaped the runtime's own handlers, an OutOfMemoryError or a StackOverflowError
 * that native image raised where no catch ran: static, so nothing is allocated to say it. */
static const char escaped_out_of_memory[] =
    "out of memory: the runtime's heap is full; raise it with -XX:MaxHeapSize=<size> through configure or "
    STR(TOKNROLL_OPTIONS_ENV) " (a vocabulary of 262k tokens needs about 256 MB)";
static const char escaped_stack_overflow[] = "stack overflow: the input nests deeper than the calling thread's stack allows";
static const char escaped_other[] = "internal error: an exception escaped the library; the call failed and the process goes on (please report it)";

/* The thread's error as the static words for kind, 1 out of memory, 2 stack overflow, anything else an escaped
 * exception: Errors.java's, for an error it caught, and every entry point's last resort for one that reached
 * the entry point past every catch, which also quarantines the call's tokenizer (escaped 1): handlers were
 * skipped, so what the call was updating may be half-written. Allocates nothing that can fail the call: the
 * message buffer is the thread's own, or the static fallback when it cannot have one. */
static const char* fail_with(const char* words) {
    char* buffer = (char*) error_buffer(0, 1);
    if (buffer) snprintf(buffer, 1025, "%s", words);
    error_code = TOKNROLL_ERROR;
    return buffer ? buffer : words;
}

const char* toknroll__escaped(int32_t kind, int32_t escaped) {
    if (escaped) escape_seen = 1;
    return fail_with(kind == 1 ? escaped_out_of_memory : kind == 2 ? escaped_stack_overflow : escaped_other);
}

/* Tokenizers an escape may have left half-updated (toknroll_shim.c): refused until freed. Checked on every
 * call, so empty, as it nearly always is, costs one load. ponytail: a scan under a lock; a hash set if a
 * process ever quarantines more than a handful. */
static port_mutex_t quarantine_lock = PORT_MUTEX_INIT;
static const void** quarantine;
static int32_t quarantined, quarantine_room;
static int quarantine_lost;   /* one could not be kept, out of C memory: every tokenizer is refused */
static long quarantine_any;   /* port_load: anything above */

int toknroll__quarantined(const void* tokenizer) {
    if (!port_load(&quarantine_any)) return 0;
    port_mutex_lock(&quarantine_lock);
    int found = quarantine_lost;
    for (int32_t i = 0; i < quarantined && !found; i++) found = quarantine[i] == tokenizer;
    port_mutex_unlock(&quarantine_lock);
    if (found) fail_with("this tokenizer may have been left half-updated by an earlier out-of-memory or stack overflow: free it and load it again");
    return found;
}

void toknroll__quarantine(const void* tokenizer) {
    port_mutex_lock(&quarantine_lock);
    if (quarantined == quarantine_room) {
        int32_t room = quarantine_room ? 2 * quarantine_room : 8;
        const void** grown = realloc((void*) quarantine, (size_t) room * sizeof *grown);
        if (grown) { quarantine = grown; quarantine_room = room; }
    }
    if (quarantined < quarantine_room) quarantine[quarantined++] = tokenizer;
    else quarantine_lost = 1;
    port_store(&quarantine_any, 1);
    port_mutex_unlock(&quarantine_lock);
}

void toknroll__release(const void* tokenizer) {
    if (!port_load(&quarantine_any)) return;
    port_mutex_lock(&quarantine_lock);
    for (int32_t i = 0; i < quarantined; i++)
        if (quarantine[i] == tokenizer) { quarantine[i] = quarantine[--quarantined]; break; }
    if (!quarantined && !quarantine_lost) port_store(&quarantine_any, 0);
    port_mutex_unlock(&quarantine_lock);
}

/* What toknroll_log_set or jinfer_log_set gave, before the runtime starts; start hands it to Log.java. */
static toknroll__log_fn log_fn;
static void* log_user_data;
static long log_given;                  /* port_load/port_store, after the two above */
static char program_property[] = "-Dtoknroll.program=" STR(TOKNROLL_PROGRAM);   /* Log.java's name, and <NAME>_LOG */
void toknroll__log_start(graal_thread* thread, int given, toknroll__log_fn fn, void* user_data);   /* Native.java */

PORT_PRINTF(2, 3) static int32_t early(int32_t code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(early_error, sizeof early_error, format, args);
    va_end(args);
    return early_code = code;
}

/* a new call: the thread's last early failure is over */
static void early_clear(void) {
    early_error[0] = '\0';
    early_code = TOKNROLL_OK;
}

/* the early error a call that finds no runtime leaves: why there is none, whatever came between */
static void no_runtime(void) {
    memcpy(early_error, start_error, sizeof early_error);
    early_code = start_code;
}

/* A thread that has called in, kept under key and freed when it exits. With the fork gate (libtoknroll on
 * POSIX), busy says it is inside a call, written with __atomic, and next lists it for fork to wait on. */
typedef struct caller {
    graal_thread* thread;
    int busy;
    struct caller* next;
} caller;

#if !defined(_WIN32) && !defined(TOKNROLL_FORK_REFUSED)
#define GATE 1
static caller* callers;          /* every thread attached */
static pthread_mutex_t callers_lock = PTHREAD_MUTEX_INITIALIZER;   /* starting, attaching, detaching; fork holds it */
static int forking;
static void gate_lock(void) { pthread_mutex_lock(&callers_lock); }
static void gate_unlock(void) { pthread_mutex_unlock(&callers_lock); }

/* The runtime forks too, to run curl: from a call in progress, or from a thread of its own. That child
 * execs at once, and the forking thread may hold what a busy thread waits for, so the gate stays out of
 * it. Each handler asks again; the answer cannot change between them. */
static int forked_by_runtime(void) {
    if (!isolate || !graal_get_current_thread(isolate)) return 0;   /* no runtime, or a thread it never saw */
    caller* c = port_tls_get(key);
    return !c || __atomic_load_n(&c->busy, __ATOMIC_RELAXED);
}

static int any_busy(void) {   /* the caller holds the gate's lock */
    for (caller* c = callers; c; c = c->next)
        if (__atomic_load_n(&c->busy, __ATOMIC_SEQ_CST)) return 1;
    return 0;
}

/* Waits for every busy thread, the lock released between looks: a busy thread may need it, as a first
 * batch does to list its workers. A thread entering meanwhile steps back, seeing forking. */
static void fork_prepare(void) {
    if (forked_by_runtime()) return;
    __atomic_store_n(&forking, 1, __ATOMIC_SEQ_CST);
    for (;;) {
        gate_lock();
        if (!any_busy()) return;   /* held until fork_done */
        gate_unlock();
        sched_yield();
    }
}

static void fork_done(void) {
    if (forked_by_runtime()) return;
    __atomic_store_n(&forking, 0, __ATOMIC_SEQ_CST);
    gate_unlock();
}
#else
static void gate_lock(void) {}
static void gate_unlock(void) {}
#endif

#if !defined(_WIN32) && defined(TOKNROLL_FORK_REFUSED)
#define REFUSE 1
static volatile int forked;      /* the child's own: written and read by its one thread */
static const char refused[] = "lib" STR(TOKNROLL_PROGRAM) " was started before this process forked, and a forked"
                              " child cannot use it: its threads stayed in the parent. Start children with spawn,"
                              " not fork";
_Static_assert(sizeof refused <= sizeof early_error, "the reason fits the early error");
/* async-signal-safe only, as a child of a threaded process must be: no stdio */
static void fork_child(void) {
    if (!isolate) return;
    memcpy(start_error, refused, sizeof refused);
    start_code = TOKNROLL_ERROR;
    memcpy(early_error, refused, sizeof refused);
    early_code = TOKNROLL_ERROR;
    forked = 1;
}
#endif

/* the record of a thread the runtime knows: listed, under key; the caller holds the gate's lock. NULL out
 * of memory, the thread detached again: attached without a record it would be outside the gate, so its
 * next call starts afresh. */
static caller* attached(graal_thread* thread) {
    caller* c = calloc(1, sizeof *c);
    if (!c) {
        graal_detach_thread(thread);
        return NULL;
    }
    c->thread = thread;
#ifdef GATE
    c->next = callers;
    callers = c;
#endif
    port_tls_set(key, c);
    return c;
}

static void detach(void* value) {
    caller* c = value;
#ifdef REFUSE
    if (forked) { free(c); return; }   /* the runtime is the parent's */
#endif
    gate_lock();
#ifdef GATE
    for (caller** at = &callers; *at; at = &(*at)->next)
        if (*at == c) { *at = c->next; break; }
#endif
    graal_detach_thread(c->thread);
    gate_unlock();
    free(c);
}

/* Once started, the library stays loaded: the runtime's own threads and every attached thread's exit
 * hook run its code, so an unload by the host (dlclose, FreeLibrary) would leave them jumping into memory
 * no longer mapped. Pinned, an unload succeeds and changes nothing, as for any library with threads. */
static void pin(void) {
#ifdef _WIN32
    HMODULE module;
    (void) GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                              (LPCSTR) &isolate, &module);
#else
    Dl_info self;
    if (dladdr((const void*) &isolate, &self)) (void) dlopen(self.dli_fname, RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE);
#endif
}

#ifdef _WIN32
/* wide in the ANSI code page, the one the runtime reads its command line in on Windows; NULL when the page
 * lacks a character (passed on, the string would be another), or out of memory */
static char* ansi(const wchar_t* wide) {
    UINT page = GetACP();
    int exact = page != CP_UTF8;   /* a UTF-8 page lacks nothing, and the call refuses the check for it */
    BOOL lost = FALSE;
    DWORD flags = exact ? WC_NO_BEST_FIT_CHARS : 0;
    int size = WideCharToMultiByte(page, flags, wide, -1, NULL, 0, NULL, exact ? &lost : NULL);
    char* s = size > 0 && !lost ? malloc((size_t) size) : NULL;
    if (s) WideCharToMultiByte(page, flags, wide, -1, s, size, NULL, NULL);
    return s;
}
#endif

/* a UTF-8 option, copied as the runtime reads it; on Windows NULL as ansi's, or when it is not UTF-8 */
static char* option_copy(const char* utf8) {
#ifdef _WIN32
    int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, NULL, 0);
    wchar_t* wide = size > 0 ? malloc((size_t) size * sizeof *wide) : NULL;
    if (!wide) return NULL;
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, wide, size);
    char* s = ansi(wide);
    free(wide);
    return s;
#else
    return port_strdup(utf8);
#endif
}

#ifdef TOKNROLL_SELF_PROPERTY
/* The path of the file this code was loaded from, the shared library or the program it is linked into,
 * found from the address of anything in it, as the runtime reads it: a copy, NULL when there is none. */
static char* self_path(void) {
#ifdef _WIN32
    wchar_t wide[4096];
    HMODULE module;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR) &isolate, &module)) return NULL;
    DWORD length = GetModuleFileNameW(module, wide, (DWORD) (sizeof wide / sizeof *wide));
    if (length == 0 || length == sizeof wide / sizeof *wide) return NULL;   /* none, or cut short */
    char* path = ansi(wide);
    if (path) return path;
    wchar_t brief[4096];   /* the page lacks one of its characters (a Chinese folder on a Latin system): 8.3, ASCII */
    length = GetShortPathNameW(wide, brief, (DWORD) (sizeof brief / sizeof *brief));
    return length && length < sizeof brief / sizeof *brief ? ansi(brief) : NULL;
#else
    Dl_info self;
    if (!dladdr((const void*) &isolate, &self)) return NULL;
    /* as the loader opened it, which is relative when the program's rpath is: the runtime wants it absolute */
    char* absolute = realpath(self.dli_fname, NULL);
    if (!absolute) return port_strdup(self.dli_fname);
    char* copy = port_strdup(absolute);
    free(absolute);
    return copy;
#endif
}
#endif

/* a start that failed: said now, and kept for every later call */
PORT_PRINTF(1, 2) static void failed(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(start_error, sizeof start_error, format, args);
    va_end(args);
    start_code = TOKNROLL_ERROR;
    no_runtime();
}

/* Error 23 in words: on x86-64, the x86-64-v3 features this CPU lacks as the process sees it, where the compiler
 * can ask (GCC, clang). A virtual machine shows its own CPU model, and QEMU's default has no AVX2. */
static void cpu_too_old(void) {
#if defined(__x86_64__) || defined(_M_X64)
    char lacks[32] = "";
#if defined(__GNUC__)
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) strcat(lacks, ", avx2");
    if (!__builtin_cpu_supports("fma")) strcat(lacks, ", fma");
    if (!__builtin_cpu_supports("bmi2")) strcat(lacks, ", bmi2");
#endif
    failed("the runtime could not be started (error 23): the library needs an x86-64-v3 CPU, with AVX2, FMA and BMI2, %s%s; "
           "in a virtual machine, pass the host's CPU through (QEMU: -cpu host)",
           *lacks ? "and this one has no " : "which this one is not", *lacks ? lacks + 2 : "");
#else
    failed("the runtime could not be started (error 23): this CPU lacks instructions the library was built for");
#endif
}

static void out_of_memory(void) { failed("the runtime could not be started: out of memory, a new process is needed"); }

static void start(void) {
    port_store(&created, 1);
    pin();
    if (port_tls_create(&key, detach) != 0) { out_of_memory(); return; }
#ifdef GATE
    if (pthread_atfork(fork_prepare, fork_done, fork_done) != 0) { out_of_memory(); return; }
#elif defined(REFUSE)
    if (pthread_atfork(NULL, NULL, fork_child) != 0) { out_of_memory(); return; }
#endif
    /* argv[0], configure's options, then the image's own file for the library linked into it, then the
     * environment, which wins: later wins as on a command line; last -Dtoknroll.program, which nothing
     * overrides. Split on whitespace; a value with a space in it goes through configure. ponytail: no quoting */
    static const char blanks[] = " \t\r\n";
    const char* e = getenv(STR(TOKNROLL_OPTIONS_ENV));   /* as the runtime reads it: on Windows, the ANSI code page */
    char* env = port_strdup(e ? e : "");   /* argv points into it */
    if (!env) { out_of_memory(); return; }
    size_t tokens = 0;
    for (const char* s = env + strspn(env, blanks); *s; s += strspn(s, blanks)) {
        tokens++;
        s += strcspn(s, blanks);
    }
    char** argv = malloc((3 + (size_t) option_count + tokens) * sizeof *argv);   /* argv[0], the image's own file, -Dtoknroll.program */
    char* self = NULL;   /* the image's own file as an option */
    if (!argv) { free(env); out_of_memory(); return; }
    int argc = 0;
    argv[argc++] = program;
    for (int i = 0; i < option_count; i++) argv[argc++] = options[i];
#ifdef TOKNROLL_SELF_PROPERTY
    char* path = self_path();
    if (path) {
        static const char prefix[] = "-D" STR(TOKNROLL_SELF_PROPERTY) "=";
        size_t size = sizeof prefix + strlen(path);
        self = malloc(size);
        if (self) snprintf(self, size, "%s%s", prefix, path);
        free(path);
        if (!self) { free(argv); free(env); out_of_memory(); return; }
        argv[argc++] = self;
    }
#endif
    for (char* s = env + strspn(env, blanks); *s; s += strspn(s, blanks)) {
        argv[argc++] = s;
        s += strcspn(s, blanks);
        if (*s) *s++ = '\0';
    }
    argv[argc++] = program_property;
    isolate_params params;
    memset(&params, 0, sizeof params);
    params.version = 5;
    params.argc = argc;
    params.argv = argv;
    graal_thread* thread;
    int rc = graal_create_isolate(&params, &isolate, &thread);
    /* the runtime parsed its command line into strings of its own while starting: the C copy goes */
    int from_env = *env != '\0';
    free(self);
    free(argv);
    free(env);
    if (rc != 0) {
        /* the C API returns a code only; the runtime names a rejected option on stderr itself. 22 is
         * ARGUMENT_PARSING_FAILED in GraalVM's CEntryPointErrors, stable across the releases we build with */
        isolate = NULL;
        const char* source = STR(TOKNROLL_OPTIONS_ENV);
        if (configured_by) source = from_env ? "configure or the environment" : configured_by;
        if (rc == 22) failed("the runtime could not be started: an option from %s was rejected, see stderr for which; a new process is needed", source);
        /* 23: CPU_FEATURE_CHECK_FAILED, the runtime's check that this CPU has what the image was built for */
        else if (rc == 23) cpu_too_old();
        /* 801 and 802: the address space the runtime reserves for the heap it may grow to, 32 GiB of
         * virtual memory, never committed, exceeds what this process may have: ulimit -v alone counts it */
        else if (rc == 801 || rc == 802) failed("the runtime could not be started (error %d): it could not reserve the address space for its heap, about 34 GB of virtual memory and not RAM, more than this process may have; raise its virtual memory limit (ulimit -v); a new process is needed", rc);
        else failed("the runtime could not be started (error %d), a new process is needed", rc);
        return;
    }
    attached(thread);
    /* running from here: a call the log callback makes now, the start's own warning calling back in, goes
     * straight to the runtime instead of into the once that is still running this */
    port_store(&started, 1);
    int given = (int) port_load(&log_given);   /* before the two it publishes */
    toknroll__log_start(thread, given, log_fn, log_user_data);   /* before any line */
}

static void init(void) {
    gate_lock();
    start();
    gate_unlock();
}

graal_thread* toknroll__thread(void) {
    early_clear();   /* every call passes here, libjinfer's and libtoknroll's: what fails below sets it again */
    if (!port_load(&started)) port_once(&once, init);
#ifdef REFUSE
    if (forked) { no_runtime(); return NULL; }
#endif
    if (!isolate) { no_runtime(); return NULL; }
    caller* c = port_tls_get(key);
    if (!c) {
        /* a thread the runtime runs itself, a log callback's on one: its own, no record, no gate */
        graal_thread* own = graal_get_current_thread(isolate);
        if (own) return own;
        graal_thread* t;
        gate_lock();
        c = graal_attach_thread(isolate, &t) == 0 ? attached(t) : NULL;   /* fails only out of memory */
        gate_unlock();
        if (!c) {
            early(TOKNROLL_ERROR, "the thread could not be attached to the runtime: out of memory");
            return NULL;
        }
    }
    return c->thread;
}

/* Every libtoknroll call is enter, the call, leave. A thread that finds a fork under way steps back and
 * waits on the lock fork holds; fork waits for every busy thread. busy counts the calls a thread is in: a
 * log callback may call in (a free, last_error) from inside one, and fork waits for the outer call anyway. */
#ifdef GATE
graal_thread* toknroll__enter(void) {
    graal_thread* t = toknroll__thread();
    if (!t) return NULL;
    caller* c = port_tls_get(key);
    if (!c) return t;   /* the runtime's own thread: fork already waits for it, as a worker or not at all */
    if (c->busy) {   /* only this thread writes it */
        __atomic_store_n(&c->busy, c->busy + 1, __ATOMIC_SEQ_CST);
        return c->thread;
    }
    for (;;) {
        __atomic_store_n(&c->busy, 1, __ATOMIC_SEQ_CST);
        if (!__atomic_load_n(&forking, __ATOMIC_SEQ_CST)) return c->thread;
        __atomic_store_n(&c->busy, 0, __ATOMIC_SEQ_CST);
        gate_lock();
        gate_unlock();
    }
}

void toknroll__leave(void) {
    caller* c = isolate ? port_tls_get(key) : NULL;
    if (c) __atomic_store_n(&c->busy, c->busy - 1, __ATOMIC_RELEASE);
}
#else
graal_thread* toknroll__enter(void) { return toknroll__thread(); }
void toknroll__leave(void) {}
#endif

/* ---- Batch workers ----
 * A batch's tasks run on its calling thread and on workers of the library's own: C threads with an 8 MiB
 * stack, started by the first parallel batch, attached to the runtime, waiting for work here, in C. A task
 * is one call of toknroll__task (Workers.java), an entry point whose exception handler returns how the task
 * ended when it ended past every catch, so C counts every task done whatever happened in it: a batch never
 * waits on a task that cannot finish, and reports an escaped one as its failure. With the fork gate a worker
 * is busy from just before it claims a task until it finishes it, so fork finds it idle, never holding a task
 * the calling thread waits on; a forked child has no workers and runs its batches on the calling thread
 * (Workers.java). A worker that cannot start or attach is not there: the batch runs on the others. */

int32_t toknroll__task(graal_thread* thread, void* tasks, int32_t task);   /* Workers.java */

typedef struct job {
    void* tasks;              /* the Java task list, a handle */
    int32_t count;
    int32_t next, done;       /* the next task to claim, and the tasks finished: atomic (port_*32) */
    int32_t escaped;          /* how a task ended past every catch (Escape's kinds), 0 for none: atomic */
    int32_t refs;             /* its caller and the workers on it, atomic: the last one frees it */
    struct job* link;         /* queued, under pool_lock */
} job;

#define WORKER_STACK ((size_t) 8 << 20)
static port_mutex_t pool_lock = PORT_MUTEX_INIT;
static port_cond_t pool_work = PORT_COND_INIT;    /* a job was queued */
static port_mutex_t done_lock = PORT_MUTEX_INIT;  /* its own: the workers just woken crowd pool_lock */
static port_cond_t pool_done = PORT_COND_INIT;    /* a job's last task finished, under done_lock */
static job* jobs;                                 /* queued, oldest first: under pool_lock */
static int pool_started;

/* the oldest job with a task left to claim; the caller holds pool_lock */
static job* claimable(void) {
    for (job* j = jobs; j; j = j->link)
        if (port_load32(&j->next) < j->count) return j;
    return NULL;
}

/* every task of j that is left, each counted done whatever happened in it */
static void work(graal_thread* thread, job* j) {
    for (int32_t task; (task = port_fetch_add32(&j->next, 1)) < j->count;) {
        int32_t escaped = toknroll__task(thread, j->tasks, task);
        if (escaped) port_set_if_zero32(&j->escaped, escaped);
        if (port_fetch_add32(&j->done, 1) + 1 == j->count) {
            port_mutex_lock(&done_lock);
            port_cond_broadcast(&pool_done);
            port_mutex_unlock(&done_lock);
        }
    }
}

PORT_THREAD(worker, number) {
    char name[32];
    snprintf(name, sizeof name, STR(TOKNROLL_PROGRAM) "-batch-%d", (int) (intptr_t) number);
    name[15] = 0;   /* Linux keeps 15 bytes and refuses more: toknroll's number does not fit */
    port_thread_name(name);
    graal_thread* thread;
    gate_lock();   /* attached and listed as a caller is: a worker lives as long as the runtime */
    caller* c = graal_attach_thread(isolate, &thread) == 0 ? attached(thread) : NULL;
    gate_unlock();
    if (!c) return PORT_THREAD_END;   /* out of memory: one worker fewer */
    port_mutex_lock(&pool_lock);
    for (;;) {
        job* j;
        while (!(j = claimable())) port_cond_wait(&pool_work, &pool_lock);
        (void) port_fetch_add32(&j->refs, 1);   /* queued, so its caller holds it too */
        port_mutex_unlock(&pool_lock);
        /* through the gate, busy until its tasks are done: it may wait for a fork here, holding the job
           but no task of it, and nothing waits for a worker */
        (void) toknroll__enter();
        work(thread, j);
        toknroll__leave();
        if (port_fetch_add32(&j->refs, -1) == 1) free(j);
        port_mutex_lock(&pool_lock);
    }
}

/* Workers.java's batch: count tasks of the task list, on self and on the pool's workers, of which the
 * first batch starts `workers`; returns how a task ended past every catch, 0 for none. Called from inside
 * an entry point, so self already holds its place in the gate. */
int32_t toknroll__batch(graal_thread* self, void* tasks, int32_t count, int32_t workers) {
    job* j = calloc(1, sizeof *j);   /* the heap's, not this stack's: a worker may be on it after the last task */
    if (!j) {   /* out of C memory: every task here */
        int32_t escaped = 0;
        for (int32_t i = 0; i < count; i++) {
            int32_t e = toknroll__task(self, tasks, i);
            if (!escaped) escaped = e;
        }
        return escaped;
    }
    j->tasks = tasks;
    j->count = count;
    j->refs = 1;
    port_mutex_lock(&pool_lock);
    if (!pool_started) {
        pool_started = 1;
        for (int32_t i = 0; i < workers; i++) (void) port_thread_start(worker, (void*) (intptr_t) (i + 1), WORKER_STACK);   /* a failed one is not there */
    }
    job** tail = &jobs;
    while (*tail) tail = &(*tail)->link;
    *tail = j;
    port_cond_broadcast(&pool_work);
    port_mutex_unlock(&pool_lock);
    work(self, j);   /* the calling thread works on its own batch too */
    port_mutex_lock(&done_lock);
    while (port_load32(&j->done) < count) port_cond_wait(&pool_done, &done_lock);
    port_mutex_unlock(&done_lock);
    port_mutex_lock(&pool_lock);
    for (job** at = &jobs; *at; at = &(*at)->link)
        if (*at == j) { *at = j->link; break; }
    port_mutex_unlock(&pool_lock);
    int32_t escaped = port_load32(&j->escaped);
    if (port_fetch_add32(&j->refs, -1) == 1) free(j);
    return escaped;
}

int toknroll__created(void) { return port_load(&created) != 0; }

/* it started, and serves this process: last_error asks it, not the early error */
static int running(void) {
#ifdef REFUSE
    if (forked) return 0;
#endif
    return port_load(&started) != 0;
}

static const char* early_last(int32_t* code) {
    if (code) *code = early_code;
    return early_error;
}

/* The calling thread's last failure: its own early one, which stands until its next call, else what the
 * runtime holds for it. Never starts the runtime just to report on it. */
const char* toknroll__last_error_of(int32_t* code, const char* (*runtime)(graal_thread*, int32_t*)) {
    if (early_error[0] || !running()) return early_last(code);
    graal_thread* t = toknroll__enter();   /* clears an early error that is empty already */
    const char* last = t ? runtime(t, code) : early_last(code);   /* NULL: the attach failed, as it says */
    toknroll__leave();
    return last;
}

#define ONCE "%s must come once, before any other call into the library"

/* configure and log_set once the runtime has started, reported through the API's own last error */
static int32_t too_late(const char* who, void (*error_set)(graal_thread*, int32_t, const char*)) {
    graal_thread* t = toknroll__enter();
    char message[128];
    snprintf(message, sizeof message, ONCE, who);
    if (t) error_set(t, TOKNROLL_EINVAL, message);
    else early(TOKNROLL_EINVAL, "%s", message);   /* no runtime to hold it: it failed to start, or a forked child */
    toknroll__leave();
    return TOKNROLL_EINVAL;
}

/* Once, before anything else: a command line is written in one place, so options from two call sites can
 * never contradict each other. The runtime starts here, so a rejected option fails this call and not some
 * later one. */
int32_t toknroll__configure(const char* const* given, int32_t count, const char* who,
                            void (*error_set)(graal_thread*, int32_t, const char*)) {
    if (toknroll__created()) return too_late(who, error_set);
    if (count < 0 || (count && !given)) return early(TOKNROLL_EINVAL, "%s: options is NULL or count negative", who);
    for (int32_t i = 0; i < count; i++) {
        if (!given[i]) return early(TOKNROLL_EINVAL, "%s: options[%d] is NULL", who, (int) i);
        if (!given[i][strspn(given[i], " \t\r\n")])   /* the runtime would drop it unsaid */
            return early(TOKNROLL_EINVAL, "%s: options[%d] is blank", who, (int) i);
    }
    char** copy = count ? calloc((size_t) count, sizeof *copy) : NULL;
    if (count && !copy) return early(TOKNROLL_ERROR, "%s: out of memory", who);
    for (int32_t i = 0; i < count; i++) {
        copy[i] = option_copy(given[i]);
        if (copy[i]) continue;
        for (int32_t j = 0; j < i; j++) free(copy[j]);
        free(copy);
#ifdef _WIN32   /* ponytail: out of memory reads as this too */
        return early(TOKNROLL_EINVAL, "%s: options[%d] is not UTF-8, or has a character the ANSI code page lacks,"
                                      " which the runtime reads its options in", who, (int) i);
#else
        return early(TOKNROLL_ERROR, "%s: out of memory", who);
#endif
    }
    options = copy;
    option_count = count;
    configured_by = who;
    int32_t rc = toknroll__enter() ? TOKNROLL_OK : TOKNROLL_ERROR;
    toknroll__leave();
    return rc;
}

/* Once, before any other call, configure included: where the lines go is fixed before the first one.
 * Never starts the runtime, so configure may follow. ponytail: like configure, not safe against a first
 * call racing it on another thread; that is the caller's ordering to keep */
int32_t toknroll__log(toknroll__log_fn fn, void* user_data, const char* who,
                      void (*error_set)(graal_thread*, int32_t, const char*)) {
    if (toknroll__created()) return too_late(who, error_set);
    if (port_load(&log_given)) return early(TOKNROLL_EINVAL, ONCE, who);
    log_fn = fn;
    log_user_data = user_data;
    port_store(&log_given, 1);
    early_clear();
    return TOKNROLL_OK;
}

/* ---- The terminal, for the hub's download board: the image's Console provider (ShimConsole.java) makes
 * the hub's few console calls here, so the image carries no Foreign Function API. fd is 0, 1 or 2. ---- */

#ifdef _WIN32
static HANDLE std_handle(int fd) { return GetStdHandle(fd == 1 ? STD_OUTPUT_HANDLE : fd == 2 ? STD_ERROR_HANDLE : STD_INPUT_HANDLE); }
static UINT code_page_before;   /* the console's output code page before toknroll__tty_utf8 set UTF-8, restored at exit */
static void restore_code_page(void) { SetConsoleOutputCP(code_page_before); }
#endif

/* 1 when fd is a terminal (Windows: a console) */
int toknroll__tty(int fd) {
#ifdef _WIN32
    DWORD mode;
    return GetConsoleMode(std_handle(fd), &mode) ? 1 : 0;
#else
    return isatty(fd) == 1;
#endif
}

/* 1 when fd is a terminal that takes ANSI sequences; on Windows the console is switched to them */
int toknroll__tty_ansi(int fd) {
#ifdef _WIN32
    DWORD mode;
    HANDLE h = std_handle(fd);
    return GetConsoleMode(h, &mode) && SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) ? 1 : 0;
#else
    return isatty(fd) == 1;
#endif
}

/* the columns of fd's terminal, 0 when it has none */
int toknroll__tty_columns(int fd) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;   /* the visible window, not the scrollback buffer */
    return GetConsoleScreenBufferInfo(std_handle(fd), &info) ? info.srWindow.Right - info.srWindow.Left + 1 : 0;
#else
    struct winsize size;
    return ioctl(fd, TIOCGWINSZ, &size) == 0 ? size.ws_col : 0;
#endif
}

/* Windows: 1 once the console's output code page is UTF-8, set here and restored at exit; elsewhere 0,
 * the locale decides */
int toknroll__tty_utf8(void) {
#ifdef _WIN32
    UINT now = GetConsoleOutputCP();
    if (!now) return 0;
    if (now == CP_UTF8) return 1;
    if (!SetConsoleOutputCP(CP_UTF8)) return 0;
    if (!code_page_before) { code_page_before = now; atexit(restore_code_page); }
    return 1;
#else
    return 0;
#endif
}
