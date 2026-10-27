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
 * where it would hang. Windows has no fork. */
#define _GNU_SOURCE               /* dladdr */
#define _DARWIN_C_SOURCE
#ifndef _WIN32
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
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
#define STR_(x) #x
#define STR(x) STR_(x)

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
static char** argv;                     /* the runtime's command line, kept for its lifetime */
static char* env;                       /* the environment's options, which argv points into */
static const char* configured_by;       /* the public function configure came through, once */
static long created;                    /* set when creation starts, port_load/port_store: then configure is too late */
static char early_error[256];           /* a failure with no isolate to hold the message: before it, or a forked child's */
static int32_t early_code;

PORT_PRINTF(2, 3) static int32_t early(int32_t code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(early_error, sizeof early_error, format, args);
    va_end(args);
    return early_code = code;
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

static void fork_prepare(void) {
    if (forked_by_runtime()) return;
    gate_lock();
    __atomic_store_n(&forking, 1, __ATOMIC_SEQ_CST);
    for (caller* c = callers; c; c = c->next)
        while (__atomic_load_n(&c->busy, __ATOMIC_SEQ_CST)) sched_yield();
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
    memcpy(early_error, refused, sizeof refused);
    early_code = TOKNROLL_ERROR;
    forked = 1;
}
#endif

/* the record of a thread the runtime knows: listed, under key; the caller holds the gate's lock. NULL out
 * of memory, and the thread stays attached without one: its next call tries again. */
static caller* attached(graal_thread* thread) {
    caller* c = calloc(1, sizeof *c);
    if (!c) return NULL;
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

#ifdef TOKNROLL_SELF_PROPERTY
/* The path of the file this code was loaded from, the shared library or the program it is linked into,
 * found from the address of anything in it. */
static const char* self_path(void) {
#ifdef _WIN32
    static char path[4096 * 3];   /* UTF-8, the encoding the runtime reads its options in; a path is wide on Windows */
    wchar_t wide[4096];
    HMODULE module;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR) &isolate, &module)) return NULL;
    if (!GetModuleFileNameW(module, wide, (DWORD) (sizeof wide / sizeof *wide))) return NULL;
    return WideCharToMultiByte(CP_UTF8, 0, wide, -1, path, (int) sizeof path, NULL, NULL) ? path : NULL;
#else
    Dl_info self;
    return dladdr((const void*) &isolate, &self) ? self.dli_fname : NULL;
#endif
}
#endif

static void out_of_memory(void) { early(TOKNROLL_ERROR, "the runtime could not be started: out of memory"); }

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
     * environment, which wins: later wins as on a command line. Split on whitespace; a value with a space in
     * it goes through configure. ponytail: no quoting */
    static const char blanks[] = " \t\r\n";
    const char* e = getenv(STR(TOKNROLL_OPTIONS_ENV));
    env = port_strdup(e ? e : "");
    if (!env) { out_of_memory(); return; }
    size_t tokens = 0;
    for (const char* s = env + strspn(env, blanks); *s; s += strspn(s, blanks)) {
        tokens++;
        s += strcspn(s, blanks);
    }
    argv = malloc((2 + (size_t) option_count + tokens) * sizeof *argv);   /* argv[0], the image's own file */
    if (!argv) { out_of_memory(); return; }
    int argc = 0;
    argv[argc++] = program;
    for (int i = 0; i < option_count; i++) argv[argc++] = options[i];
#ifdef TOKNROLL_SELF_PROPERTY
    const char* self = self_path();
    if (self) {
        static const char prefix[] = "-D" STR(TOKNROLL_SELF_PROPERTY) "=";
        size_t size = sizeof prefix + strlen(self);
        char* option = malloc(size);
        if (!option) { out_of_memory(); return; }
        snprintf(option, size, "%s%s", prefix, self);
        argv[argc++] = option;
    }
#endif
    for (char* s = env + strspn(env, blanks); *s; s += strspn(s, blanks)) {
        argv[argc++] = s;
        s += strcspn(s, blanks);
        if (*s) *s++ = '\0';
    }
    isolate_params params;
    memset(&params, 0, sizeof params);
    params.version = 5;
    params.argc = argc;
    params.argv = argv;
    graal_thread* thread;
    int rc = graal_create_isolate(&params, &isolate, &thread);
    if (rc != 0) {
        /* the C API returns a code only; the runtime names a rejected option on stderr itself. 22 is
         * ARGUMENT_PARSING_FAILED in GraalVM's CEntryPointErrors, stable across the releases we build with */
        isolate = NULL;
        const char* source = STR(TOKNROLL_OPTIONS_ENV);
        if (configured_by) source = *env ? "configure or the environment" : configured_by;
        if (rc == 22) early(TOKNROLL_ERROR, "the runtime could not be started: an option from %s was rejected, see stderr for which", source);
        else early(TOKNROLL_ERROR, "the runtime could not be started (error %d)", rc);
        return;
    }
    attached(thread);
}

static void init(void) {
    gate_lock();
    start();
    gate_unlock();
}

graal_thread* toknroll__thread(void) {
    port_once(&once, init);
#ifdef REFUSE
    if (forked) return NULL;
#endif
    if (!isolate) return NULL;
    caller* c = port_tls_get(key);
    if (!c) {
        graal_thread* t;
        gate_lock();
        c = graal_attach_thread(isolate, &t) == 0 ? attached(t) : NULL;   /* ponytail: unreported; fails only out of memory */
        gate_unlock();
        if (!c) return NULL;
    }
    return c->thread;
}

/* Every libtoknroll call is enter, the call, leave. A thread that finds a fork under way steps back and
 * waits on the lock fork holds; fork waits for every busy thread. */
#ifdef GATE
graal_thread* toknroll__enter(void) {
    if (!toknroll__thread()) return NULL;
    caller* c = port_tls_get(key);
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
    if (c) __atomic_store_n(&c->busy, 0, __ATOMIC_RELEASE);
}
#else
graal_thread* toknroll__enter(void) { return toknroll__thread(); }
void toknroll__leave(void) {}
#endif

int toknroll__created(void) { return port_load(&created) != 0; }

const char* toknroll__early_error(int32_t* code) {
    if (code) *code = early_code;
    return early_error;
}

/* Once, before anything else: a command line is written in one place, so options from two call sites can
 * never contradict each other. The runtime starts here, so a rejected option fails this call and not some
 * later one. */
int32_t toknroll__configure(const char* const* given, int32_t count, const char* who,
                            void (*error_set)(graal_thread*, int32_t, const char*)) {
    if (toknroll__created()) {
        graal_thread* t = toknroll__enter();
        char message[128];
        snprintf(message, sizeof message, "%s must come once, before any other call into the library", who);
        if (t) error_set(t, TOKNROLL_EINVAL, message);
        toknroll__leave();
        return TOKNROLL_EINVAL;
    }
    if (count < 0 || (count && !given)) return early(TOKNROLL_EINVAL, "%s: options is NULL or count negative", who);
    for (int32_t i = 0; i < count; i++)
        if (!given[i]) return early(TOKNROLL_EINVAL, "%s: options[%d] is NULL", who, (int) i);
    char** copy = count ? calloc((size_t) count, sizeof *copy) : NULL;
    if (count && !copy) return early(TOKNROLL_ERROR, "%s: out of memory", who);
    for (int32_t i = 0; i < count; i++) {
        copy[i] = port_strdup(given[i]);
        if (!copy[i]) {
            while (i--) free(copy[i]);
            free(copy);
            return early(TOKNROLL_ERROR, "%s: out of memory", who);
        }
    }
    options = copy;
    option_count = count;
    configured_by = who;
    int32_t rc = toknroll__enter() ? TOKNROLL_OK : TOKNROLL_ERROR;
    toknroll__leave();
    return rc;
}
