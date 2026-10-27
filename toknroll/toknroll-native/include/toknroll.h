/* toknroll.h - token-perfect LLM tokenization for any language. C99, one header, no dependencies.
 *
 *   Errors    Pointer-returning functions return NULL, int32_t-returning functions return a
 *             negative TOKNROLL_E* code. toknroll_last_error() has the message and, when asked,
 *             the code, per calling thread, until its next toknroll call; after one that succeeds,
 *             "" and TOKNROLL_OK. Reading does not clear it.
 *   Buffers   (out, capacity) follows snprintf: at most capacity elements are written, the
 *             number needed is returned, strings get a NUL when there is room. Capacity 0 sizes.
 *   Strings   UTF-8. Text and token spellings come with their length in bytes, so any byte may be
 *             in them; with length 0 the pointer may be NULL. Sources and options end at their NUL.
 *             The one string returned to you, toknroll_last_error's, is the calling thread's until
 *             its next toknroll call. Nothing is freed but tokenizers, by toknroll_free, which
 *             accepts NULL.
 *   Threads   Any thread may call, and a tokenizer may be shared between threads: it never changes.
 *             A process may fork at any time; the child keeps every tokenizer it inherited.
 *   Runtime   The library is a Java runtime compiled ahead of time; toknroll_configure passes it
 *             options as a java command line would, before the first call, and the TOKNROLL_OPTIONS
 *             environment variable adds to or overrides them. Nothing requires either. Once
 *             started the library stays loaded, as its threads run its code: an unload (dlclose,
 *             FreeLibrary) succeeds and leaves it in place.
 *   Sources   A path, or a reference to a hub. Spelling tells them apart: /x, C:\x, ./x, ../x, ~/x,
 *             a backslash, owner/file.gguf or a name that exists on disk is a path; anything else
 *             is a reference, [host/]owner/repo[@revision][/path][:quant], the host hf.co (the
 *             default), huggingface.co or modelscope.cn. To force the hub, write the host; to force
 *             the disk, write ./ What is reached decides the format: a file by its content, the
 *             GGUF magic or a JSON tokenizer.json, whatever its name; a folder by name, its
 *             tokenizer.json. A bare owner/repo is the repository root, a folder: it never opens a
 *             GGUF. A GGUF on a hub, named by :quant or a /path ending in .gguf, is read for its
 *             header only, never its weights. Files land in the cache jinfer shares;
 *             -Dtoknroll.cache.root moves it, -Dtoknroll.offline=true never fetches. Missing is
 *             TOKNROLL_ENOTFOUND; a malformed reference, an unknown host, a file named twice (by
 *             /path and :quant) or a file of no known kind is TOKNROLL_EINVAL.
 *               Qwen/Qwen3-0.6B                         the repository's tokenizer.json
 *               LiquidAI/LFM2.5-350M-GGUF:Q8_0          the header of that GGUF
 *               org/repo/tokenizers/v2                  a folder inside a repository
 *               modelscope.cn/Qwen/Qwen3-0.6B@v1        the other hub, a pinned revision
 *               ~/.ollama/models/blobs/sha256-3f1a...   a GGUF with no extension
 *   Stability MAJOR breaks the ABI, MINOR only adds, PATCH only fixes; 0.x is held to the same
 *             rule as 1.x. A shipped signature, name or default never changes. toknroll__* is
 *             private.
 */
#ifndef TOKNROLL_H
#define TOKNROLL_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32) && defined(TOKNROLL_BUILD)
#  define TOKNROLL_API __declspec(dllexport)
#elif defined(_WIN32)
#  define TOKNROLL_API __declspec(dllimport)
#elif defined(__GNUC__)
#  define TOKNROLL_API __attribute__((visibility("default")))
#else
#  define TOKNROLL_API
#endif

#define TOKNROLL_VERSION_MAJOR 0
#define TOKNROLL_VERSION_MINOR 3
#define TOKNROLL_VERSION_PATCH 1

#define TOKNROLL_OK         0
#define TOKNROLL_ERROR     (-1) /* see toknroll_last_error() */
#define TOKNROLL_EINVAL    (-2)
#define TOKNROLL_ENOTFOUND (-3) /* file, repository or tokenizer */

/* Token flags, a 64-bit mask: bit n is GGUF token type n, exactly one of these six is set; higher
 * bits are reserved for flags to come. Control and user-defined are the special tokens. */
#define TOKNROLL_TOKEN_NORMAL       (INT64_C(1) << 1)
#define TOKNROLL_TOKEN_UNKNOWN      (INT64_C(1) << 2)
#define TOKNROLL_TOKEN_CONTROL      (INT64_C(1) << 3)
#define TOKNROLL_TOKEN_USER_DEFINED (INT64_C(1) << 4)
#define TOKNROLL_TOKEN_UNUSED       (INT64_C(1) << 5)
#define TOKNROLL_TOKEN_BYTE         (INT64_C(1) << 6)

typedef struct toknroll_tokenizer toknroll_tokenizer;

TOKNROLL_API const char* toknroll_version(void);                /* "0.3.1", the toknroll release it was built from */
TOKNROLL_API const char* toknroll_last_error(int32_t* code);    /* the message; *code = its code unless NULL, TOKNROLL_OK after success */
/* Runtime options, as on a java command line: "-Dtoknroll.gguf.pre.my-model=alias:qwen2" registers a
 * pre-tokenizer, "-Dtoknroll.cache.root=/srv/cache" moves the hub cache, "-XX:MaxHeapSize=512m" caps the
 * heap. Once, before any other call into the library; a later call is TOKNROLL_EINVAL. An option the
 * runtime rejects fails this call, TOKNROLL_ERROR, the runtime naming it on stderr (an unknown -XX:
 * option is rejected, an unknown -X one is ignored). TOKNROLL_OPTIONS in the environment, the same
 * options separated by whitespace, is read when the runtime starts and applied after these, so it
 * overrides them: the program sets its options, whoever runs it has the last word. */
TOKNROLL_API int32_t toknroll_configure(const char* const* options, int32_t count);

/* A tokenizer from a path or a hub reference: Sources, above. */
TOKNROLL_API toknroll_tokenizer* toknroll_load(const char* source);
TOKNROLL_API void                toknroll_free(toknroll_tokenizer* t);

/* The ids of length bytes of text. specials: tokens of a type other than normal spelled in the text
 * stay whole; by default they are text. */
TOKNROLL_API int32_t toknroll_encode(const toknroll_tokenizer* t, const char* text, size_t length,
                                     bool specials, int32_t* out, int32_t capacity);
/* The tokens' bytes: UTF-8, except where a token holds part of a character. An id outside the
 * vocabulary is TOKNROLL_EINVAL. */
TOKNROLL_API int32_t toknroll_decode(const toknroll_tokenizer* t, const int32_t* ids, int32_t count,
                                     char* out, int32_t capacity);
/* count texts in one buffer back to back, text i ending at byte text_ends[i]; their ids back to back
 * in out, text i's ending at ends[i]; encoded on all cores. Returns the total, capacity 0 counts. */
TOKNROLL_API int32_t toknroll_encode_batch(const toknroll_tokenizer* t, const char* texts,
                                           const int32_t* text_ends, int32_t count, bool specials,
                                           int32_t* out, int32_t capacity, int32_t* ends);

TOKNROLL_API int32_t toknroll_vocabulary_size(const toknroll_tokenizer* t);
/* An id's TOKNROLL_TOKEN_* flags; negative, TOKNROLL_ENOTFOUND, when there is no such id. */
TOKNROLL_API int64_t toknroll_token_flags(const toknroll_tokenizer* t, int32_t id);
/* The spelling of an id, as the vocabulary writes it; TOKNROLL_ENOTFOUND when there is no such id. */
TOKNROLL_API int32_t toknroll_token(const toknroll_tokenizer* t, int32_t id, char* out, int32_t capacity);
/* The id of a spelling, length bytes of it; TOKNROLL_ENOTFOUND when the vocabulary has no such token. */
TOKNROLL_API int32_t toknroll_id(const toknroll_tokenizer* t, const char* token, size_t length);

#ifdef __cplusplus
}
#endif
#endif
