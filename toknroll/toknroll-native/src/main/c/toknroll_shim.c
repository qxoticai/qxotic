/* The C face of libtoknroll: forwards each header function to the toknroll__* entry point the image
 * exports, which takes the isolate thread first. The isolate is toknroll_isolate.c's. */
#include <stddef.h>
#include "toknroll.h"

typedef struct _graal_isolatethread_t graal_isolatethread_t;
graal_isolatethread_t* toknroll__thread(void);               /* toknroll_isolate.c */
int toknroll__created(void);
const char* toknroll__early_error(int32_t* code);
int32_t toknroll__configure(const char* const*, int32_t, const char* who, void (*error_set)(graal_isolatethread_t*, int32_t, const char*));
void toknroll__error_set(graal_isolatethread_t*, int32_t, const char*);

typedef graal_isolatethread_t* T;
#define THREAD T t = toknroll__thread()
#define STR(x) #x
#define XSTR(x) STR(x)

const char* toknroll__last_error(T, int32_t*);
toknroll_tokenizer* toknroll__load(T, const char*);
void toknroll__free(T, toknroll_tokenizer*);
int32_t toknroll__encode(T, const toknroll_tokenizer*, const char*, bool, int32_t*, int32_t);
int32_t toknroll__decode(T, const toknroll_tokenizer*, const int32_t*, int32_t, char*, int32_t);
int32_t toknroll__encode_batch(T, const toknroll_tokenizer*, const char*, const int32_t*, int32_t, bool, int32_t*, int32_t, int32_t*);
int32_t toknroll__vocabulary_size(T, const toknroll_tokenizer*);
int64_t toknroll__token_flags(T, const toknroll_tokenizer*, int32_t);
int32_t toknroll__token(T, const toknroll_tokenizer*, int32_t, char*, int32_t);
int32_t toknroll__id(T, const toknroll_tokenizer*, const char*);

const char* toknroll_version(void) { return XSTR(TOKNROLL_VERSION_MAJOR) "." XSTR(TOKNROLL_VERSION_MINOR) "." XSTR(TOKNROLL_VERSION_PATCH); }
const char* toknroll_last_error(int32_t* code) {
    T t = toknroll__created() ? toknroll__thread() : NULL;   /* never create the isolate just to report on it */
    return t ? toknroll__last_error(t, code) : toknroll__early_error(code);
}
int32_t toknroll_configure(const char* const* options, int32_t count) { return toknroll__configure(options, count, "toknroll_configure", toknroll__error_set); }

toknroll_tokenizer* toknroll_load(const char* source) { THREAD; return t ? toknroll__load(t, source) : NULL; }
void toknroll_free(toknroll_tokenizer* tk) { if (!tk) return; THREAD; if (t) toknroll__free(t, tk); }   /* NULL never starts the runtime */

int32_t toknroll_encode(const toknroll_tokenizer* tk, const char* text, bool specials, int32_t* out, int32_t cap) { THREAD; return t ? toknroll__encode(t, tk, text, specials, out, cap) : TOKNROLL_ERROR; }
int32_t toknroll_decode(const toknroll_tokenizer* tk, const int32_t* ids, int32_t count, char* out, int32_t cap) { THREAD; return t ? toknroll__decode(t, tk, ids, count, out, cap) : TOKNROLL_ERROR; }

int32_t toknroll_encode_batch(const toknroll_tokenizer* tk, const char* texts, const int32_t* text_ends, int32_t count, bool specials, int32_t* out, int32_t cap, int32_t* ends) { THREAD; return t ? toknroll__encode_batch(t, tk, texts, text_ends, count, specials, out, cap, ends) : TOKNROLL_ERROR; }

int32_t toknroll_vocabulary_size(const toknroll_tokenizer* tk) { THREAD; return t ? toknroll__vocabulary_size(t, tk) : TOKNROLL_ERROR; }
int64_t toknroll_token_flags(const toknroll_tokenizer* tk, int32_t id) { THREAD; return t ? toknroll__token_flags(t, tk, id) : TOKNROLL_ERROR; }
int32_t toknroll_token(const toknroll_tokenizer* tk, int32_t id, char* out, int32_t cap) { THREAD; return t ? toknroll__token(t, tk, id, out, cap) : TOKNROLL_ERROR; }
int32_t toknroll_id(const toknroll_tokenizer* tk, const char* token) { THREAD; return t ? toknroll__id(t, tk, token) : TOKNROLL_ERROR; }
