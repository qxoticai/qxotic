/* The C tests' few helpers, libtoknroll's and libjinfer's: checks, a process per scenario, stderr
 * captured, threads, files; POSIX and Windows.
 *
 * A library reads its settings once, as its runtime starts, so each scenario runs in a fresh process:
 * `test` runs them all, a child each; `test <name>` runs one, here, its variable set first. Checks print
 * to stdout, so a scenario may capture stderr whole. */
#ifndef TEST_H
#define TEST_H
#if defined(_WIN32)
#  define _CRT_SECURE_NO_WARNINGS     /* fopen, getenv, freopen: the standard ones, not MSVC's _s variants */
#  define _CRT_NONSTDC_NO_WARNINGS
#elif !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L     /* setenv, fork, alarm: before any system header */
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#  include <direct.h>
#  include <process.h>
#  include <windows.h>
#  define NL "\r\n"                   /* the end of a line the library writes: Java's line separator */
#else
#  include <pthread.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  define NL "\n"
#endif

#define COUNT(a) ((int) (sizeof (a) / sizeof *(a)))

/* ---- checks: each failure says where, and what it saw ---- */

static int failures;
#define FAIL_AT(...) (printf("    %s:%d: ", __FILE__, __LINE__), printf(__VA_ARGS__), printf("\n"), failures++)

#define CHECK(cond) do { if (!(cond)) FAIL_AT("%s", #cond); } while (0)
#define CHECK_EQ(got, want) do { long long g_ = (long long) (got), w_ = (long long) (want); \
    if (g_ != w_) FAIL_AT("%s is %lld, want %lld", #got, g_, w_); } while (0)
#define CHECK_STR(got, want) do { const char *g_ = (got), *w_ = (want); \
    if (!g_ || strcmp(g_, w_)) FAIL_AT("%s is \"%s\", want \"%s\"", #got, g_ ? g_ : "(null)", w_); } while (0)
#define CHECK_HAS(got, part) do { const char *g_ = (got), *p_ = (part); \
    if (!g_ || !strstr(g_, p_)) FAIL_AT("%s is \"%s\", which lacks \"%s\"", #got, g_ ? g_ : "(null)", p_); } while (0)

/* A call returned code, and the thread's last error says so: the same code, a message holding part ("" after
 * a success). last_error is the library's own: toknroll_last_error, jinfer_last_error. */
#define CHECK_ERROR(got, code, last_error, part) do { long long g_ = (long long) (got); int32_t c_ = 99; \
    const char* m_ = last_error(&c_); \
    if (g_ != (code) || c_ != (code) || !m_ || !strstr(m_, part) || ((code) == 0 && *m_)) \
        FAIL_AT("%s is %lld, last error %d \"%s\"; want %d, \"%s\"", #got, g_, (int) c_, m_ ? m_ : "(null)", (int) (code), part); \
    } while (0)
#define CHECK_NULL_ERROR(got, code, last_error, part) CHECK_ERROR((got) ? 1 : (code), code, last_error, part)   /* 1: not NULL */
#define CHECK_LAST(code, last_error, part) CHECK_ERROR(code, code, last_error, part)   /* a call whose return cannot say */
#define CHECK_CLEAR(last_error) CHECK_ERROR(0, 0, last_error, "")

/* ---- files, under target/ctest ---- */

#define TMP(name) "target/ctest/" name

static void make_dir(const char* path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static void write_file(const char* path, const void* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("    cannot write %s\n", path); exit(2); }
    fwrite(data, 1, size, f);
    fclose(f);
}

/* stderr into a file, the runtime's own writes included; captured_stderr reads it back, the caller frees */
static void capture_stderr(void) {
    if (!freopen(TMP("stderr.txt"), "wb", stderr)) { printf("    cannot capture stderr\n"); exit(2); }
}

static char* captured_stderr(void) {
    fflush(stderr);
    FILE* f = fopen(TMP("stderr.txt"), "rb");
    char* text = (char*) calloc(1, 1 << 20);
    if (f) { (void) !fread(text, 1, (1 << 20) - 1, f); fclose(f); }   /* ponytail: a MiB of stderr is plenty */
    return text;
}

/* ---- threads ---- */

typedef void (*thread_fn)(void* arg);
typedef struct { thread_fn fn; void* arg; } thread_start;
#ifdef _WIN32
typedef HANDLE thread_t;
static DWORD WINAPI thread_main(LPVOID p) { thread_start s = *(thread_start*) p; free(p); s.fn(s.arg); return 0; }
#else
typedef pthread_t thread_t;
static void* thread_main(void* p) { thread_start s = *(thread_start*) p; free(p); s.fn(s.arg); return NULL; }
#endif

static thread_t thread_run(thread_fn fn, void* arg) {
    thread_start* s = (thread_start*) malloc(sizeof *s);
    s->fn = fn;
    s->arg = arg;
#ifdef _WIN32
    return CreateThread(NULL, 0, thread_main, s, 0, NULL);
#else
    pthread_t t;
    pthread_create(&t, NULL, thread_main, s);
    return t;
#endif
}

static void thread_join(thread_t t) {
#ifdef _WIN32
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
#else
    pthread_join(t, NULL);
#endif
}

/* ---- scenarios ---- */

typedef struct {
    const char* name;
    void (*run)(void);
    const char* env;   /* "NAME=value" set before it runs, or NULL */
} scenario;

static void env_set(const char* name, const char* value) {   /* NULL removes it */
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

/* none of the libraries' variables the shell may have set, but the scenario's own */
static void scenario_env(const scenario* s) {
    static const char* const libraries[] = {"TOKNROLL_LOG", "TOKNROLL_OPTIONS", "TOKNROLL_CACHE_ROOT", "JINFER_LOG",
                                            "JINFER_OPTIONS"};
    for (int k = 0; k < COUNT(libraries); k++) env_set(libraries[k], NULL);
    if (s->env) {
        char name[64];
        snprintf(name, sizeof name, "%.*s", (int) (strchr(s->env, '=') - s->env), s->env);
        env_set(name, strchr(s->env, '=') + 1);
    }
}

/* main: every scenario, a child each, or the one named here. The parent sets a child's variables too, so
 * they are in its environment from its start: a DLL's own C runtime copies the environment as it loads. */
static int test_main(int argc, char** argv, const scenario* scenarios, int count) {
    setvbuf(stdout, NULL, _IONBF, 0);
    make_dir("target/ctest");
    if (argc > 1) {
        for (int i = 0; i < count; i++) {
            if (strcmp(argv[1], scenarios[i].name)) continue;
            scenario_env(&scenarios[i]);
            scenarios[i].run();
            return failures ? 1 : 0;
        }
        printf("no scenario %s\n", argv[1]);
        return 2;
    }
    int failed = 0;
    for (int i = 0; i < count; i++) {
        scenario_env(&scenarios[i]);
#ifdef _WIN32
        const char* args[] = {argv[0], scenarios[i].name, NULL};
        int ok = _spawnv(_P_WAIT, argv[0], args) == 0;
#else
        pid_t pid = fork();
        if (pid == 0) { execl(argv[0], argv[0], scenarios[i].name, (char*) NULL); _exit(127); }
        int status = 0;
        waitpid(pid, &status, 0);
        int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (WIFSIGNALED(status)) printf("    killed by signal %d\n", WTERMSIG(status));
#endif
        printf("%s %s\n", ok ? "ok  " : "FAIL", scenarios[i].name);
        failed += !ok;
    }
    printf("%d of %d scenarios passed\n", count - failed, count);
    return failed ? 1 : 0;
}

#endif
