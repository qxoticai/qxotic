/* tokenize SOURCE TEXT: ids, the text back, the vocabulary size. cc -Iinclude tokenize.c -ltoknroll */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toknroll.h"

static int fail(const char* what) {
    int32_t code = 0;
    fprintf(stderr, "%s: %s (%" PRId32 ")\n", what, toknroll_last_error(&code), code);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: tokenize SOURCE TEXT\n"); return 2; }
    toknroll_tokenizer* t = toknroll_load(argv[1]);
    if (!t) return fail("load");

    size_t length = strlen(argv[2]);
    int32_t n = toknroll_encode(t, argv[2], length, false, NULL, 0);  /* capacity 0 counts */
    if (n < 0) return fail("encode");
    int32_t* ids = malloc((size_t) n * sizeof *ids);
    if (n && !ids) { fprintf(stderr, "out of memory\n"); return 1; }
    toknroll_encode(t, argv[2], length, false, ids, n);
    for (int32_t i = 0; i < n; i++) printf("%" PRId32 "%s", ids[i], i + 1 < n ? " " : "\n");

    char text[1024];
    if (toknroll_decode(t, ids, n, text, sizeof text) < 0) return fail("decode");
    printf("%s\n%" PRId32 " tokens, vocabulary %" PRId32 ", toknroll %s\n", text, n, toknroll_vocabulary_size(t), toknroll_version());

    free(ids);
    toknroll_free(t);
    return 0;
}
