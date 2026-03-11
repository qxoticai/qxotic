#ifndef JOTA_MOJO_H
#define JOTA_MOJO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JOTA_MOJO_ABI_VERSION 1

typedef enum jota_mojo_status {
  JOTA_MOJO_STATUS_OK = 0,
  JOTA_MOJO_STATUS_INVALID_ARGUMENT = 1,
  JOTA_MOJO_STATUS_RUNTIME_ERROR = 2
} jota_mojo_status;

typedef uint64_t jota_mojo_context_handle;

typedef struct jota_mojo_init_options {
  uint32_t abi_version;
  const char *fixed_target;
  const char *fixed_backend;
} jota_mojo_init_options;

jota_mojo_status jota_mojo_context_init(const jota_mojo_init_options *options,
                                        jota_mojo_context_handle *out_context);

jota_mojo_status jota_mojo_context_shutdown(jota_mojo_context_handle context);

const char *jota_mojo_context_last_error(jota_mojo_context_handle context);

#ifdef __cplusplus
}
#endif

#endif
