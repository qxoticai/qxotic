/* libjinfer from C: load, chat, stream, free. cc -Iinclude -I<toknroll>/include chat.c -ljinfer */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "jinfer.h"

/* The library is silent until asked: this prints everything it says. A program keeps what it wants
 * by level: downloads and warnings are INFO and above, loads and kernel choices DEBUG. */
static void log_line(jinfer_log_level level, const char* message, void* payload) {
    (void) level; (void) payload;
    fprintf(stderr, "%s\n", message);
}

int main(int argc, char** argv) {
    const char* ref = argc > 1 ? argv[1] : "LiquidAI/LFM2.5-350M-GGUF:Q8_0";
    const char* request = argc > 2 ? argv[2]   /* a chat.completions request; its text is JSON text, escaped */
        : "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One sentence.\"}],\"max_tokens\":64,\"temperature\":0}";
    printf("libjinfer %s\n", jinfer_version());
    jinfer_log_set(log_line, NULL);

    jinfer_model* model = jinfer_model_load(ref, NULL);
    if (!model) { fprintf(stderr, "load: %s\n", jinfer_last_error(NULL)); return 1; }
    printf("context %" PRId32 ", vocab %" PRId32 "\n", jinfer_model_property(model, JINFER_CONTEXT_CAPACITY), jinfer_model_property(model, JINFER_VOCAB_SIZE));

    jinfer_stream* stream = jinfer_chat(model, request);
    if (!stream) { fprintf(stderr, "chat: %s\n", jinfer_last_error(NULL)); return 1; }

    int32_t rc;
    while ((rc = jinfer_stream_step(stream)) == JINFER_OK) {
        printf("%s", jinfer_stream_text(stream));
        fflush(stdout);
    }
    printf("\n");
    if (rc < 0) { fprintf(stderr, "step: %s\n", jinfer_last_error(NULL)); return 1; }
    printf("%s\n", jinfer_stream_result(stream));

    jinfer_stream_free(stream);
    jinfer_model_free(model);
    return 0;
}
