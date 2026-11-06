/* A download's progress, as the C API's host sees it: on unless the host says off, plain lines on stderr
 * where there is no terminal, the live board on one; off, nothing. The download is Qwen's tokenizer.json
 * (11 MB) over an empty cache root, so the network; without it the scenario skips, which fails the run as a
 * missing fixture does. Included by toknroll_test.c after PROGRAM and QWEN are defined. */
#ifdef TEST_LIBJINFER
#  define PROGRESS_VAR "JINFER_PROGRESS"
#else
#  define PROGRESS_VAR "TOKNROLL_PROGRESS"
#endif

#ifdef _WIN32
#  include <process.h>
#  define getpid _getpid
#else
#  ifdef __APPLE__
#    include <util.h>
#  else
#    include <pty.h>
#  endif
#  include <errno.h>
#  include <sys/ioctl.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

static void progress_remove(const char* dir) {
    char cmd[700];
#ifdef _WIN32
    snprintf(cmd, sizeof cmd, "rmdir /s /q \"%s\" 2>nul", dir);
#else
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
#endif
    (void) system(cmd);
}

/* the file fetched into a cache of this process alone, the store's root and the HF cache under one folder,
 * removed before and after */
static void progress_download(const char* option) {
    static char dir[512], root[600], hf[600];
    snprintf(dir, sizeof dir, TMP("progress-%ld"), (long) getpid());
    progress_remove(dir);
    snprintf(root, sizeof root, "-D" PROGRAM ".cache.root=%s/store", dir);
    snprintf(hf, sizeof hf, "%s/hf", dir);
    env_set("HF_HOME", hf);   /* the hub's write-through goes there, not to this machine's cache */
    const char* options[] = {root, option};
    CHECK_EQ(toknroll_configure(options, option ? 2 : 1), TOKNROLL_OK);
    toknroll_tokenizer* t = toknroll_tokenizer_load(QWEN);
    if (!t) skip("%s does not download (offline?): %s", QWEN, toknroll_last_error(NULL));
    toknroll_tokenizer_free(t);
    progress_remove(dir);
}

static void progress_on_by_default(void) {   /* nothing said, no terminal: plain lines, no escape sequence */
    capture_stderr();
    progress_download(NULL);
    char* err = captured_stderr();
    CHECK_HAS(err, "download hf.co/" QWEN);
    CHECK_HAS(err, "tokenizer.json  ");
    CHECK_HAS(err, "%  ");   /* a tenth */
    CHECK_HAS(err, " done in ");
    CHECK(!strstr(err, "\x1b["));
    CHECK(!strstr(err, PROGRAM ": "));   /* progress, not log lines */
    free(err);
}

static void progress_off(void) {   /* PROGRESS_VAR=off: nothing on stderr; at the default log level nothing at all */
    capture_stderr();
    progress_download(NULL);
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void progress_property_over_the_environment(void) {   /* PROGRESS_VAR=on, the option off: off */
    capture_stderr();
    progress_download("-D" PROGRAM ".progress=off");
    char* err = captured_stderr();
    CHECK_STR(err, "");
    free(err);
}

static void progress_board(void) {   /* the download alone, stderr as it is: progress_on_a_terminal's child, run by name only */
    progress_download(NULL);
}

#ifndef _WIN32
static void progress_on_a_terminal(void) {   /* nothing said, stderr a pseudo-terminal: the live board */
    int master = -1;
    struct winsize size = {24, 100, 0, 0};
    pid_t pid = forkpty(&master, NULL, NULL, &size);
    if (pid < 0) skip("no pseudo-terminal: %s", strerror(errno));
    if (pid == 0) {
        env_set("TERM", "xterm");
        env_set("NO_COLOR", NULL);
        execl(test_self, test_self, ".progress_board", (char*) NULL);
        _exit(127);
    }
    char* out = (char*) calloc(1, 1 << 20);
    size_t have = 0;
    for (;;) {
        ssize_t n = read(master, out + have, (1 << 20) - 1 - have);
        if (n <= 0) break;   /* EIO at the child's end */
        have += (size_t) n;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    close(master);
    CHECK_EQ(WIFEXITED(status) ? WEXITSTATUS(status) : -1, 0);
    CHECK_HAS(out, "download hf.co/" QWEN);
    CHECK_HAS(out, "\x1b[2K");   /* a repainted row */
    CHECK_HAS(out, "tokenizer.json");
    CHECK(!strstr(out, "tokenizer.json  "));   /* no plain line, label then two spaces: the board's rows only */
    free(out);
}
#  define PROGRESS_TERMINAL_SCENARIO {"progress_on_a_terminal", progress_on_a_terminal, NULL},
#else
#  define PROGRESS_TERMINAL_SCENARIO
#endif

#define PROGRESS_SCENARIOS \
    {"progress_on_by_default", progress_on_by_default, NULL}, \
    {"progress_off", progress_off, PROGRESS_VAR "=off"}, \
    {"progress_property_over_the_environment", progress_property_over_the_environment, PROGRESS_VAR "=on"}, \
    {".progress_board", progress_board, NULL}, \
    PROGRESS_TERMINAL_SCENARIO
