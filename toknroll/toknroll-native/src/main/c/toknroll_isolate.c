/* The isolate behind the toknroll__* entry points: created by configure, or else on first use, with the
 * options configure passed plus those in the environment, each calling thread attached once and detached
 * when it exits. libjinfer compiles this file too (TOKNROLL_PROGRAM "jinfer", TOKNROLL_OPTIONS_ENV
 * "JINFER_OPTIONS", TOKNROLL_SELF_PROPERTY "jam.native.library.path"), so its jinfer_* and toknroll_*
 * symbols share one isolate and one configure, and libjam, linked into it, is loaded from the image. */
#define _GNU_SOURCE               /* dladdr */
#define _DARWIN_C_SOURCE
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toknroll.h"
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

typedef struct _graal_isolate_t graal_isolate_t;
typedef struct _graal_isolatethread_t graal_isolatethread_t;
/* graal_isolate.h's graal_create_isolate_params_t, version 5, as the generated header documents it:
 * argv is "parsed like command line arguments" (argv[0] is the program), and unrecognized arguments
 * fail creation unless ignored (they are not: a typo must fail, not do nothing). Spelled out here
 * because the header exists only once the image is built, after this file is compiled. */
typedef struct {
    int version;
    uintptr_t reserved_address_space_size;
    const char* auxiliary_image_path;             /* internal */
    uintptr_t auxiliary_image_reserved_space_size; /* internal */
    int argc;
    char** argv;
    int pkey;                                     /* internal */
    char ignore_unrecognized_args;
    char _reserved_4;
    char _reserved_5;
} isolate_params;
int graal_create_isolate(isolate_params* params, graal_isolate_t** isolate, graal_isolatethread_t** thread);
int graal_attach_thread(graal_isolate_t* isolate, graal_isolatethread_t** thread);
int graal_detach_thread(graal_isolatethread_t* thread);

static graal_isolate_t* isolate;
static port_once_t once = PORT_ONCE_INIT;
static port_tls_t key;
static char* program = STR(TOKNROLL_PROGRAM);    /* argv[0]; GraalVM parses the rest like a java command line */
static char** argv = &program;
static int argc = 1;
static const char* configured_by;          /* the public function configure came through, once */
static volatile int created;           /* set when creation starts: from then on configure is too late */
static char early_error[256];          /* a failure before there was an isolate to hold the message */
static int32_t early_code;

static int32_t early(int32_t code, const char* format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(early_error, sizeof early_error, format, args);
    va_end(args);
    return early_code = code;
}

static void detach(void* thread) { graal_detach_thread((graal_isolatethread_t*) thread); }

#ifdef TOKNROLL_SELF_PROPERTY
/* The path of the file this code was loaded from: the shared library, or the program it is linked into. */
static const char* self_path(void) {
#ifdef _WIN32
    static char path[4096 * 3];   /* UTF-8, the encoding the runtime reads its options in; a path is wide on Windows */
    wchar_t wide[4096];
    HMODULE module;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR) &self_path, &module)) return NULL;
    if (!GetModuleFileNameW(module, wide, sizeof wide / sizeof *wide)) return NULL;
    return WideCharToMultiByte(CP_UTF8, 0, wide, -1, path, sizeof path, NULL, NULL) ? path : NULL;
#else
    Dl_info self;
    return dladdr((void*) &self_path, &self) ? self.dli_fname : NULL;
#endif
}
#endif

static void init(void) {
    created = 1;
    /* configure's options first, then the image's own file for the library linked into it, then the
     * environment, which wins: later wins as on a command line. Split on whitespace; a value with a space in
     * it goes through configure. ponytail: no quoting */
    const char* e = getenv(STR(TOKNROLL_OPTIONS_ENV));
    char* env = port_strdup(e ? e : "");
    char** all = malloc((argc + 2 + strlen(env) / 2) * sizeof *all);   /* a token needs a char and a gap */
    memcpy(all, argv, argc * sizeof *argv);
    argv = all;
#ifdef TOKNROLL_SELF_PROPERTY
    const char* self = self_path();
    if (self) {
        char* option = malloc(sizeof "-D" STR(TOKNROLL_SELF_PROPERTY) "=" + strlen(self));
        strcpy(option, "-D" STR(TOKNROLL_SELF_PROPERTY) "=");
        argv[argc++] = strcat(option, self);
    }
#endif
    for (char* tok = strtok(env, " \t\n"); tok; tok = strtok(NULL, " \t\n")) argv[argc++] = tok;
    isolate_params params;
    memset(&params, 0, sizeof params);
    params.version = 5;
    params.argc = argc;
    params.argv = argv;
    graal_isolatethread_t* thread;
    int rc = graal_create_isolate(&params, &isolate, &thread);
    if (rc != 0) {
        /* the C API returns a code only; the runtime names a rejected option on stderr itself. 22 is
         * ARGUMENT_PARSING_FAILED in GraalVM's CEntryPointErrors, stable across the releases we build with */
        isolate = NULL;
        const char* source = STR(TOKNROLL_OPTIONS_ENV);
        if (configured_by) source = env && *env ? "configure or the environment" : configured_by;
        if (rc == 22) early(TOKNROLL_ERROR, "the runtime could not be started: an option from %s was rejected, see stderr for which", source);
        else early(TOKNROLL_ERROR, "the runtime could not be started (error %d)", rc);
        return;
    }
    port_tls_create(&key, detach);
    port_tls_set(key, thread);
}

graal_isolatethread_t* toknroll__thread(void) {
    port_once(&once, init);
    if (!isolate) return NULL;
    graal_isolatethread_t* t = port_tls_get(key);
    if (!t) {
        if (graal_attach_thread(isolate, &t) != 0) return NULL;   /* ponytail: unreported; attaching fails only out of memory */
        port_tls_set(key, t);
    }
    return t;
}

int toknroll__created(void) { return created; }

const char* toknroll__early_error(int32_t* code) {
    if (code) *code = early_code;
    return early_error;
}

/* who is the public function, for messages; error_set the API's own last-error setter (jinfer__error_set,
 * toknroll__error_set), since each API reports its own last error. Once, before anything else: a command
 * line is written in one place, so options from two call sites can never contradict each other. The
 * runtime starts here, so a rejected option fails this call and not some later one. */
int32_t toknroll__configure(const char* const* options, int32_t count, const char* who,
                            void (*error_set)(graal_isolatethread_t*, int32_t, const char*)) {
    if (created) {
        graal_isolatethread_t* t = toknroll__thread();
        char message[128];
        snprintf(message, sizeof message, "%s must come once, before any other call into the library", who);
        if (t) error_set(t, TOKNROLL_EINVAL, message);
        return TOKNROLL_EINVAL;
    }
    if (count < 0 || (count && !options)) return early(TOKNROLL_EINVAL, "%s: options is NULL or count negative", who);
    for (int32_t i = 0; i < count; i++)
        if (!options[i]) return early(TOKNROLL_EINVAL, "%s: options[%d] is NULL", who, i);
    argv = malloc((count + 1) * sizeof *argv);
    argv[0] = program;
    for (int32_t i = 0; i < count; i++) argv[i + 1] = port_strdup(options[i]);
    argc = count + 1;
    configured_by = who;
    return toknroll__thread() ? TOKNROLL_OK : TOKNROLL_ERROR;
}
