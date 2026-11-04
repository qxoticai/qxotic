// toknroll.h from C++: it compiles warning-free under -Wpedantic, its names link unmangled, and the
// values a binding hard-codes are the ones it says.
#include "toknroll.h"
#include <string>

static_assert(TOKNROLL_OK == 0 && TOKNROLL_ERROR == -1 && TOKNROLL_EINVAL == -2 && TOKNROLL_ENOTFOUND == -3, "codes");
static_assert(TOKNROLL_SPECIALS == 1u, "flags");
static_assert(TOKNROLL_TOKEN_NORMAL == 2 && TOKNROLL_TOKEN_BYTE == 64, "token flags");
static_assert(TOKNROLL_LOG_DEBUG == 0 && TOKNROLL_LOG_ERROR == 3, "log levels");

int main() {
    toknroll_log_fn fn = nullptr;
    (void) fn;
    std::string want = std::to_string(TOKNROLL_VERSION_MAJOR) + "." + std::to_string(TOKNROLL_VERSION_MINOR) + "."
                       + std::to_string(TOKNROLL_VERSION_PATCH);
    return want == toknroll_version() ? 0 : 1;
}
