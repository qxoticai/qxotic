// Build: cc -O2 -o llama_tokenize llama_tokenize.c -I$LLAMA/include -I$LLAMA/ggml/include \
//            -L$LLAMA/build/bin -lllama -Wl,-rpath,$LLAMA/build/bin
// Pure tokenization with libllama, as test-tokenizer-0 does it (add_special and parse_special
// false), one vocabulary load for many texts. stdin: per text, "<length>\n<bytes>"; stdout: one
// line of ids per text, "ERR" if llama.cpp refuses it.
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
static void quiet(enum ggml_log_level level, const char * text, void * data) { (void) level; (void) text; (void) data; }
int main(int argc, char ** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s model.gguf < texts\n", argv[0]); return 1; }
    llama_log_set(quiet, NULL);
    llama_backend_init();
    struct llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    struct llama_model * model = llama_model_load_from_file(argv[1], params);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    long length;
    while (scanf("%ld", &length) == 1) {
        getchar();
        char * text = malloc(length + 1);
        if (fread(text, 1, length, stdin) != (size_t) length) return 2;
        int capacity = (int) length * 4 + 16;
        llama_token * ids = malloc(sizeof(llama_token) * capacity);
        int n = llama_tokenize(vocab, text, (int32_t) length, ids, capacity, false, false);
        if (n < 0) { puts("ERR"); } else {
            for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
            putchar('\n');
        }
        free(ids); free(text);
    }
    llama_model_free(model);
    return 0;
}
