// jinfer.h, and the toknroll.h it includes, from C++: warning-free under -Wpedantic, unmangled at link
// time, and the values a binding hard-codes are the ones it says.
#include "jinfer.h"
#include <string>

static_assert(JINFER_OK == 0 && JINFER_DONE == 1 && JINFER_EBUSY == -6, "codes");
static_assert(JINFER_VOCAB_SIZE == 1 && JINFER_HAS_CHAT == 100 && JINFER_HAS_TRANSCRIPTION == 105, "keys");
static_assert(JINFER_LOG_DEBUG == 0 && JINFER_LOG_ERROR == 3, "log levels");
static_assert(JINFER_CONTEXT_MAXIMUM == -1, "the maximum context, -1 on every face");

int main() {
    jinfer_model_options options;
    jinfer_model_options_init(&options);
    std::string jinfer = std::to_string(JINFER_VERSION_MAJOR) + "." + std::to_string(JINFER_VERSION_MINOR) + "."
                         + std::to_string(JINFER_VERSION_PATCH);
    std::string toknroll = std::to_string(TOKNROLL_VERSION_MAJOR) + "." + std::to_string(TOKNROLL_VERSION_MINOR) + "."
                           + std::to_string(TOKNROLL_VERSION_PATCH);
    return jinfer == jinfer_version() && toknroll == toknroll_version() && options.struct_size == (int64_t) sizeof options ? 0 : 1;
}
