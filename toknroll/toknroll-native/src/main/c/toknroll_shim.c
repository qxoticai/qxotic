/* The C face of libtoknroll: forwards each header function to the toknroll__* entry point the image
 * exports, which takes the isolate thread first. The isolate, and the gate every call holds against fork,
 * are toknroll_isolate.c's. */
#include <stddef.h>
#include "toknroll.h"
#include "toknroll_isolate.h"

_Static_assert(sizeof(void*) == 8 && sizeof(size_t) == 8, "the C API assumes a 64-bit platform");

typedef graal_thread* T;
void toknroll__error_set(T, int32_t, const char*);
/* the body of a forwarding function: the call inside the gate, or fail when there is no runtime */
#define CALL(type, fail, call) { T t = toknroll__enter(); type r = t ? (call) : (fail); toknroll__leave(); return r; }
#define CALL_VOID(call) { T t = toknroll__enter(); if (t) (call); toknroll__leave(); }
#define STR(x) #x
#define XSTR(x) STR(x)

const char* toknroll__last_error(T, int32_t*);
toknroll_tokenizer* toknroll__tokenizer_load(T, const char*);
void toknroll__tokenizer_free(T, toknroll_tokenizer*);
int32_t toknroll__encode(T, const toknroll_tokenizer*, const char*, size_t, uint32_t, int32_t*, int32_t);
int64_t toknroll__decode(T, const toknroll_tokenizer*, const int32_t*, int32_t, char*, int64_t);
int32_t toknroll__encode_batch(T, const toknroll_tokenizer*, const char*, const int64_t*, int32_t, uint32_t, int32_t*, int32_t, int32_t*);
int32_t toknroll__vocabulary_size(T, const toknroll_tokenizer*);
int64_t toknroll__token_flags(T, const toknroll_tokenizer*, int32_t);
int64_t toknroll__token(T, const toknroll_tokenizer*, int32_t, char*, int64_t);
int32_t toknroll__id(T, const toknroll_tokenizer*, const char*, size_t);

const char* toknroll_version(void) { return XSTR(TOKNROLL_VERSION_MAJOR) "." XSTR(TOKNROLL_VERSION_MINOR) "." XSTR(TOKNROLL_VERSION_PATCH); }
const char* toknroll_last_error(int32_t* code) {
    if (!toknroll__created()) return toknroll__early_error(code);   /* never create the isolate just to report on it */
    CALL(const char*, toknroll__early_error(code), toknroll__last_error(t, code))
}
int32_t toknroll_configure(const char* const* options, int32_t count) { return toknroll__configure(options, count, "toknroll_configure", toknroll__error_set); }

toknroll_tokenizer* toknroll_tokenizer_load(const char* source) CALL(toknroll_tokenizer*, NULL, toknroll__tokenizer_load(t, source))
void toknroll_tokenizer_free(toknroll_tokenizer* tk) { if (tk) CALL_VOID(toknroll__tokenizer_free(t, tk)) }   /* NULL never starts the runtime */

int32_t toknroll_encode(const toknroll_tokenizer* tk, const char* text, size_t length, uint32_t flags, int32_t* out, int32_t cap) CALL(int32_t, TOKNROLL_ERROR, toknroll__encode(t, tk, text, length, flags, out, cap))
int64_t toknroll_decode(const toknroll_tokenizer* tk, const int32_t* ids, int32_t count, char* out, int64_t cap) CALL(int64_t, TOKNROLL_ERROR, toknroll__decode(t, tk, ids, count, out, cap))

int32_t toknroll_encode_batch(const toknroll_tokenizer* tk, const char* texts, const int64_t* text_offsets, int32_t count, uint32_t flags, int32_t* out, int32_t cap, int32_t* offsets) CALL(int32_t, TOKNROLL_ERROR, toknroll__encode_batch(t, tk, texts, text_offsets, count, flags, out, cap, offsets))

int32_t toknroll_vocabulary_size(const toknroll_tokenizer* tk) CALL(int32_t, TOKNROLL_ERROR, toknroll__vocabulary_size(t, tk))
int64_t toknroll_token_flags(const toknroll_tokenizer* tk, int32_t id) CALL(int64_t, TOKNROLL_ERROR, toknroll__token_flags(t, tk, id))
int64_t toknroll_token(const toknroll_tokenizer* tk, int32_t id, char* out, int64_t cap) CALL(int64_t, TOKNROLL_ERROR, toknroll__token(t, tk, id, out, cap))
int32_t toknroll_id(const toknroll_tokenizer* tk, const char* token, size_t length) CALL(int32_t, TOKNROLL_ERROR, toknroll__id(t, tk, token, length))
