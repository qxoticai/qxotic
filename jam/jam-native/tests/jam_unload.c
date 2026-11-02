/* An unloaded libjam leaves nothing behind that a later fork would run: the library pins itself, so
 * it stays mapped after dlclose and a fork afterwards is safe. Run as jam_unload <path to libjam>. */
#include <dlfcn.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: jam_unload <libjam>\n"); return 2; }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    const char* (*version)(void) = (const char* (*)(void)) dlsym(lib, "jam_version");
    if (!version || !*version()) { fprintf(stderr, "no jam_version\n"); return 1; }
    dlclose(lib);
    int failed = 0;
    if (!dlopen(argv[1], RTLD_NOW | RTLD_NOLOAD)) { printf("  [FAIL] unload: dlclose unmapped libjam\n"); failed = 1; }
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) _exit(0);
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        printf("  [FAIL] unload: a fork after dlclose ended with status %d\n", status);
        failed = 1;
    }
    if (!failed) printf("unload: libjam stays mapped after dlclose, and a fork afterwards is safe\n");
    return failed;
}
