/* tokenize SOURCE TEXT: ids, the text back, the vocabulary size. cc -Iinclude tokenize.c -ltoknroll */
#include <stdio.h>
#include <stdlib.h>
#include "toknroll.h"

static int fail(const char* what) {
    int32_t code = 0;
    fprintf(stderr, "%s: %s (%d)\n", what, toknroll_last_error(&code), code);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: tokenize SOURCE TEXT\n"); return 2; }
    toknroll_tokenizer* t = toknroll_load(argv[1]);
    if (!t) return fail("load");

    int32_t n = toknroll_encode(t, argv[2], false, NULL, 0);          /* capacity 0 counts */
    if (n < 0) return fail("encode");
    int32_t* ids = malloc(n * sizeof *ids);
    toknroll_encode(t, argv[2], false, ids, n);
    for (int32_t i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? " " : "\n");

    char text[1024];
    if (toknroll_decode(t, ids, n, text, sizeof text) < 0) return fail("decode");
    printf("%s\n%d tokens, vocabulary %d, toknroll %s\n", text, n, toknroll_vocabulary_size(t), toknroll_version());

    free(ids);
    toknroll_free(t);
    return 0;
}
