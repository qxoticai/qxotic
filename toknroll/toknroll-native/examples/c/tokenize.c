/* tokenize SOURCE TEXT: ids, the text back, the vocabulary size. cc -Iinclude tokenize.c -ltoknroll */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toknroll.h"

static int fail(const char* what) {
    int32_t code = 0;
    const char* message = toknroll_last_error(&code);
    fprintf(stderr, "%s: %s (%" PRId32 ")\n", what, message, code);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: tokenize SOURCE TEXT\n"); return 2; }
    toknroll_tokenizer* t = toknroll_tokenizer_load(argv[1]);
    if (!t) return fail("load");

    int result = 1;
    int32_t* ids = NULL;
    char* text = NULL;
    size_t length = strlen(argv[2]);
    int32_t n = toknroll_encode(t, argv[2], length, 0, NULL, 0);  /* capacity 0 counts */
    if (n < 0) { fail("encode"); goto done; }
    ids = malloc((size_t) n * sizeof *ids);
    if (n && !ids) { fprintf(stderr, "out of memory\n"); goto done; }
    if (toknroll_encode(t, argv[2], length, 0, ids, n) < 0) { fail("encode"); goto done; }

    int64_t bytes = toknroll_decode(t, ids, n, 0, NULL, 0, NULL);
    if (bytes < 0) { fail("decode"); goto done; }
    text = malloc((size_t) bytes + 1);  /* the text and its terminating NUL */
    if (!text) { fprintf(stderr, "out of memory\n"); goto done; }
    if (toknroll_decode(t, ids, n, 0, text, bytes + 1, NULL) < 0) { fail("decode"); goto done; }
    int32_t vocabulary = toknroll_vocabulary_size(t);
    if (vocabulary < 0) { fail("vocabulary"); goto done; }

    for (int32_t i = 0; i < n; i++) printf("%s%" PRId32, i ? " " : "", ids[i]);
    printf("\n%s\n%" PRId32 " tokens, vocabulary %" PRId32 ", toknroll %s\n", text, n, vocabulary, toknroll_version());
    result = 0;

done:
    free(text);
    free(ids);
    toknroll_tokenizer_free(t);
    return result;
}
