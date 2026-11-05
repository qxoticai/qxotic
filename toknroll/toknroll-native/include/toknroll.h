/* toknroll.h - token-perfect LLM tokenization for any language. C99, one header, no dependencies.
 *
 *   Errors     A function returns its result; a negative value is a TOKNROLL_E* code, NULL for a
 *              pointer. toknroll_last_error has the message and the code, per thread, until the
 *              thread's next toknroll call (reading it changes nothing); after a success, "" and
 *              TOKNROLL_OK. Every call sets it but toknroll_version, toknroll_last_error and a free of NULL.
 *              A goroutine or a Java virtual thread may change OS thread between two calls, so it
 *              reads the message pinned to the thread that failed (runtime.LockOSThread) or goes by
 *              the returned code, which is always the call's own.
 *   Buffers    (out, token_capacity) and (out, byte_capacity) follow snprintf: at most that many
 *              ids or bytes are written and the number needed is returned; capacity 0 sizes. Text
 *              is NUL-terminated whenever byte_capacity is not 0, cut within byte_capacity - 1
 *              bytes when longer, so it is whole when the return is below byte_capacity. Ids are
 *              counted in int32_t, bytes in int64_t; a negative count, length or capacity is
 *              TOKNROLL_EINVAL.
 *   Text       UTF-8. Text and token spellings come with their length in bytes, so any byte may be in
 *              them, a NUL included; with length 0 the pointer may be NULL. A malformed sequence reads
 *              as U+FFFD, as a UTF-8 decoder reads it. Sources and options end at their NUL.
 *   Ownership  What you pass is borrowed for the call; no buffer is kept. A tokenizer is yours until
 *              toknroll_tokenizer_free, which accepts NULL. The one string returned to you,
 *              toknroll_last_error's, is the calling thread's until its next toknroll call.
 *   Threads    Any thread may call, and a tokenizer may be shared between threads: it never changes.
 *              toknroll_tokenizer_free is the exception: not during another call on the same tokenizer.
 *   Fork       A process may fork at any time; the child keeps every tokenizer it inherited.
 *   Settings   The library is a Java runtime compiled ahead of time. toknroll_configure passes it
 *              options as a java command line would, before the first call, and the TOKNROLL_OPTIONS
 *              environment variable adds to or overrides them; nothing requires either. Once started
 *              it stays loaded, as its threads run its code: an unload (dlclose, FreeLibrary)
 *              succeeds and leaves it in place.
 *   Logging    TOKNROLL_LOG picks the lines: off, error, warn (the default), info or debug. They go
 *              to stderr, "toknroll: info: jinfer.hub: download ...", or to toknroll_log_set's
 *              callback. From info on, a download at a terminal draws its progress on stderr instead
 *              of logging it. Beyond them the library writes nothing to stdout or stderr, but for
 *              the runtime naming an option it rejects.
 *   Sources    A path, or a reference to a hub, in UTF-8: a file whose name is not UTF-8 cannot be
 *              named. Spelling tells them apart: /x, C:\x, ./x, ../x, ~/x (~ is not expanded, as fopen
 *              does not), a backslash, owner/file.gguf or a name that exists on disk is a path;
 *              anything else is a reference, [host/]owner/repo[@revision][/path][:quant], the host
 *              hf.co (the default), huggingface.co or modelscope.cn. To force the hub, write the host;
 *              to force the disk, write ./ What is reached decides the format: a file by its content,
 *              the GGUF magic or a JSON tokenizer.json, whatever its name; a folder by name, its
 *              tokenizer.json. A bare owner/repo is the repository root, a folder: it never opens a
 *              GGUF. A GGUF on a hub, named by :quant or a /path ending in .gguf, is read for its
 *              header only, never its weights. Files land in the cache jinfer shares;
 *              -Dtoknroll.cache.root, else TOKNROLL_CACHE_ROOT in the environment, moves it,
 *              -Dtoknroll.offline=true never fetches. Missing is TOKNROLL_ENOTFOUND; a malformed
 *              reference, an unknown host, a file named twice (by /path and :quant), a file of no
 *              known kind or one that does not parse is TOKNROLL_EINVAL. A valid one toknroll does not
 *              implement (the README lists the families it tests) is TOKNROLL_EUNSUPPORTED.
 *                Qwen/Qwen3-0.6B                         the repository's tokenizer.json
 *                LiquidAI/LFM2.5-350M-GGUF:Q8_0          the header of that GGUF
 *                org/repo/tokenizers/v2                  a folder inside a repository
 *                modelscope.cn/Qwen/Qwen3-0.6B@v1        the other hub, a pinned revision
 *                /srv/ollama/blobs/sha256-3f1a...        a GGUF with no extension
 *   Stability  Before 1.0 any release may break this API. From 1.0 a PATCH changes nothing a caller
 *              relies on, a MINOR only adds, a MAJOR may break, and every release is checked.
 */
#ifndef TOKNROLL_H
#define TOKNROLL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(TOKNROLL_STATIC)
#  define TOKNROLL_API
#elif defined(_WIN32) && defined(TOKNROLL_BUILD)
#  define TOKNROLL_API __declspec(dllexport)
#elif defined(_WIN32)
#  define TOKNROLL_API __declspec(dllimport)
#elif defined(__GNUC__)
#  define TOKNROLL_API __attribute__((visibility("default")))
#else
#  define TOKNROLL_API
#endif

#define TOKNROLL_VERSION_MAJOR 0
#define TOKNROLL_VERSION_MINOR 4
#define TOKNROLL_VERSION_PATCH 0

#define TOKNROLL_OK            0
#define TOKNROLL_ERROR        (-1) /* see toknroll_last_error() */
#define TOKNROLL_EINVAL       (-2)
#define TOKNROLL_ENOTFOUND    (-3) /* file, repository or tokenizer */
#define TOKNROLL_EUNSUPPORTED (-4) /* a valid tokenizer toknroll does not implement */

/* Token flags, a 64-bit mask: bit n is GGUF token type n, exactly one of these six is set; higher
 * bits are reserved for flags to come. Control and user-defined are the special tokens. */
#define TOKNROLL_TOKEN_NORMAL       (INT64_C(1) << 1)
#define TOKNROLL_TOKEN_UNKNOWN      (INT64_C(1) << 2)
#define TOKNROLL_TOKEN_CONTROL      (INT64_C(1) << 3)
#define TOKNROLL_TOKEN_USER_DEFINED (INT64_C(1) << 4)
#define TOKNROLL_TOKEN_UNUSED       (INT64_C(1) << 5)
#define TOKNROLL_TOKEN_BYTE         (INT64_C(1) << 6)

typedef struct toknroll_tokenizer toknroll_tokenizer;

enum { TOKNROLL_LOG_DEBUG = 0, TOKNROLL_LOG_INFO = 1, TOKNROLL_LOG_WARN = 2, TOKNROLL_LOG_ERROR = 3 };   /* a line's level */
typedef void (*toknroll_log_fn)(int32_t level, const char* message, void* user_data);

/* "0.4.0", the toknroll release the library was built from. */
TOKNROLL_API const char* toknroll_version(void);
/* The message of the thread's last call; *code = its code unless code is NULL. "" and TOKNROLL_OK after a
 * success. */
TOKNROLL_API const char* toknroll_last_error(int32_t* code);
/* The library's log lines go to fn instead of stderr, with user_data as its last argument; with fn NULL
 * they go nowhere, whatever TOKNROLL_LOG says. A line is UTF-8, without a newline, and names what wrote
 * it: "jinfer.hub: download hf.co/...". The callback runs on any thread, inside your own call included,
 * and may call only toknroll_last_error and toknroll_tokenizer_free, on a tokenizer no call is using; it
 * and user_data stay for the life of the process. Once, before any other call, toknroll_configure
 * included; a later call is TOKNROLL_EINVAL. */
TOKNROLL_API int32_t toknroll_log_set(toknroll_log_fn fn, void* user_data);
/* Runtime options, as on a java command line: "-Dtoknroll.gguf.pre.my-model=alias:qwen2" registers a
 * pre-tokenizer, "-Dtoknroll.cache.root=/srv/cache" moves the hub cache, "-XX:MaxHeapSize=512m" caps the
 * heap. Once, before any other call into the library; a later call is TOKNROLL_EINVAL. An option the
 * runtime rejects fails this call, TOKNROLL_ERROR, the runtime naming it on stderr (an unknown -XX:
 * option is rejected, an unknown -X one is ignored). TOKNROLL_OPTIONS in the environment, the same
 * options separated by whitespace, is read when the runtime starts and applied after these, so it
 * overrides them: the program sets its options, whoever runs it has the last word. */
TOKNROLL_API int32_t toknroll_configure(const char* const* options, int32_t count);

/* A tokenizer from a path or a hub reference: Sources, above. */
TOKNROLL_API toknroll_tokenizer* toknroll_tokenizer_load(const char* source);
TOKNROLL_API void                toknroll_tokenizer_free(toknroll_tokenizer* t);

/* Options for toknroll_encode and toknroll_encode_batch, or'ed together; 0 is the default. An
 * unknown bit is TOKNROLL_EINVAL. */
#define TOKNROLL_SPECIALS 1u /* tokens of a type other than normal, spelled in the text, stay whole */

/* The ids of length bytes of text; by default every token is text. */
TOKNROLL_API int32_t toknroll_encode(const toknroll_tokenizer* t, const char* text, int64_t length,
                                     uint32_t flags, int32_t* out, int32_t token_capacity);
/* The bytes ids[start, count) have in the whole text ids[0, count): UTF-8, except where a token
 * holds part of a character. A tokenizer that adds a space before a text (SentencePiece) drops it
 * from id 0 only, so a start past 0 continues the text, spaces and all, where ids + start would
 * begin a new one. Only ids[start, count) are read. Over capacity, out holds the tokens that fit
 * whole in byte_capacity - 1 bytes, then the NUL. decoded_tokens, unless NULL, gets how many from
 * start, 0 when byte_capacity is 0: start + *decoded_tokens is count exactly when the text is
 * whole. A start outside [0, count] or an id no token has is TOKNROLL_EINVAL, and decoded_tokens
 * is left as it was. */
TOKNROLL_API int64_t toknroll_decode(const toknroll_tokenizer* t, const int32_t* ids, int32_t count,
                                     int32_t start, char* out, int64_t byte_capacity,
                                     int32_t* decoded_tokens);
/* count texts, text i the bytes [text_offsets[i], text_offsets[i + 1]) of texts; their ids go to
 * out, text i's at [offsets[i], offsets[i + 1]), offsets[0] being 0. Both offset arrays hold
 * count + 1 entries. Encoded on all cores, a forked child's on the calling thread. Returns the
 * total, offsets[count]; over capacity, out gets a prefix and offsets is still complete, so
 * token_capacity 0 counts. */
TOKNROLL_API int32_t toknroll_encode_batch(const toknroll_tokenizer* t, const char* texts,
                                           const int64_t* text_offsets, int32_t count, uint32_t flags,
                                           int32_t* out, int32_t token_capacity, int32_t* offsets);

/* One past the largest id. An id below it may still have no token, a gap the file leaves:
 * toknroll_token and toknroll_token_flags say TOKNROLL_ENOTFOUND for it, and decode refuses it. */
TOKNROLL_API int32_t toknroll_vocabulary_size(const toknroll_tokenizer* t);
/* An id's TOKNROLL_TOKEN_* flags; negative, TOKNROLL_ENOTFOUND, when there is no such id. */
TOKNROLL_API int64_t toknroll_token_flags(const toknroll_tokenizer* t, int32_t id);
/* The spelling of an id: a token of a type other than normal is its text, as TOKNROLL_SPECIALS matches
 * it, any other as the vocabulary writes it; TOKNROLL_ENOTFOUND when there is no such id. */
TOKNROLL_API int64_t toknroll_token(const toknroll_tokenizer* t, int32_t id, char* out, int64_t byte_capacity);
/* The id of a spelling, length bytes of it, either kind toknroll_token gives; TOKNROLL_ENOTFOUND when the
 * vocabulary has no such token. */
TOKNROLL_API int32_t toknroll_id(const toknroll_tokenizer* t, const char* token, int64_t length);

#ifdef __cplusplus
}
#endif
#endif
