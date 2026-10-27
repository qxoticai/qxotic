/* jinfer.h - local AI inference for any language. C99, one header, no dependencies.
 *
 *   Errors    Pointer-returning functions return NULL, int32_t-returning functions return a
 *             negative JINFER_E* code. jinfer_last_error() has the message and, when asked,
 *             the code, per calling thread, until its next jinfer call; reading does not clear it.
 *   Buffers   (out, capacity) follows snprintf: at most capacity elements are written, the
 *             number needed is returned, strings get a NUL when there is room. Capacity 0
 *             sizes. Element counts are int32_t, text lengths in bytes are size_t. Strings are
 *             UTF-8, NUL-terminated unless a length is given. Sizes known in advance come from
 *             jinfer_model_property.
 *   Ownership Everything returned is borrowed from the handle it came from: a model's strings
 *             live as long as the model, a state's or a stream's until the next call on that
 *             handle. Nothing is freed but handles, each by its own _free, which accepts NULL.
 *             A model stays alive while any state or stream made from it lives; _free drops
 *             your reference.
 *   Threads   Any thread may call. A model may be shared. A state or a stream is used by
 *             one thread at a time; jinfer_stream_cancel is the exception, any thread may
 *             call it. The log callback runs on any thread, inside your own call included,
 *             and must not call jinfer. A download shows its progress on stderr when that is
 *             a terminal, as log lines otherwise. A child forked once the library has started
 *             cannot use it, the toknroll_* functions it carries included: its threads stayed in
 *             the parent, so calls fail there, JINFER_ERROR, and a _free does nothing. Start
 *             children with spawn or exec. Once started the library stays loaded, as its threads
 *             run its code: an unload (dlclose, FreeLibrary) succeeds and leaves it in place.
 *   Defaults  A request field left out takes the model file's value, listed by
 *             jinfer_model_info, never the library's.
 *   Downloads run through curl on the PATH: the library carries no TLS stack of its own.
 *             -Djinfer.offline=true never downloads; local files need nothing.
 *   Stability MAJOR breaks the ABI, MINOR only adds, PATCH only fixes; 0.x is held to the
 *             same rule as 1.x. A shipped signature, name or default never changes. New
 *             capabilities arrive as new functions and new jinfer_property keys; options
 *             structs grow at the end behind struct_size, and a zero struct_size is refused.
 *             jinfer_version keeps its name and signature in every version. jinfer__* is
 *             private.
 *   Payloads  Requests are the OpenAI JSON shapes. jinfer_chat takes and returns exactly the
 *             jinfer server's /v1/chat/completions request and response, tools, response_format
 *             with a JSON schema and media as data URLs included; jinfer_speak takes the
 *             /v1/audio/speech fields that apply (input, speed). Text inside a request is JSON
 *             text: escape it. A schema-constrained reply cannot come back malformed: tokens
 *             outside the schema are never sampled.
 */
#ifndef JINFER_H
#define JINFER_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "toknroll.h"   /* the tokenizer API: jinfer_model_tokenizer hands out a toknroll_tokenizer */
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32) && defined(JINFER_BUILD)
#  define JINFER_API __declspec(dllexport)
#elif defined(_WIN32)
#  define JINFER_API __declspec(dllimport)
#elif defined(__GNUC__)
#  define JINFER_API __attribute__((visibility("default")))
#else
#  define JINFER_API
#endif

#define JINFER_VERSION_MAJOR 0
#define JINFER_VERSION_MINOR 3
#define JINFER_VERSION_PATCH 1

#define JINFER_OK            0
#define JINFER_DONE          1    /* a stream has nothing more to produce */
#define JINFER_ERROR        (-1) /* see jinfer_last_error() */
#define JINFER_EINVAL       (-2)
#define JINFER_ENOTFOUND    (-3) /* model, file or companion */
#define JINFER_EUNSUPPORTED (-4) /* the model lacks this capability */
#define JINFER_ECANCELLED   (-5)

typedef enum { JINFER_LOG_DEBUG, JINFER_LOG_INFO, JINFER_LOG_WARN, JINFER_LOG_ERROR } jinfer_log_level;
typedef void (*jinfer_log_fn)(jinfer_log_level level, const char* message, void* payload);

JINFER_API const char* jinfer_version(void);          /* "0.3.1", the jinfer release it was built from */
JINFER_API const char* jinfer_last_error(int32_t* code);   /* the message; *code = its code unless NULL, JINFER_OK after success */
JINFER_API void        jinfer_log_set(jinfer_log_fn fn, void* payload);   /* NULL = silent, the default */
/* Runtime options, as on a java command line: "-Djinfer.threads=8" sizes the process's pool, "-Djinfer.cache.sessions=16"
 * keeps that many conversations warm (4), "-Djinfer.cache.budget=8g" bounds the prompt cache's KV blocks (2g, 0 = off),
 * "-Djinfer.offline=true" never downloads, "-Djinfer.trace=true", "-Djam.*" and the other jinfer.* properties steer the
 * engine, "-XX:MaxHeapSize=4g" caps the heap. Once, before any other call into the library; a later call is
 * JINFER_EINVAL. toknroll_configure is the same call on the same runtime. An option the runtime rejects fails this call,
 * JINFER_ERROR, the runtime naming it on stderr (an unknown -XX: option is rejected, an unknown -X one is ignored).
 * JINFER_OPTIONS in the environment, the same options separated by whitespace, is read when the runtime starts and
 * applied after these, so it overrides them: the program sets its options, whoever runs it has the last word.
 * jinfer_log_set may precede it. */
JINFER_API int32_t     jinfer_configure(const char* const* options, int32_t count);

/* ------------------------------------------------------------------ model */
/* A loaded model: the weights and configuration, plus the states that jinfer_chat, jinfer_embed and
 * jinfer_rerank run in, each holding up to context_capacity tokens of context. */
typedef struct jinfer_model jinfer_model;

typedef struct {
    size_t             struct_size;   /* set by jinfer_model_options_init */
    const char* const* companions;    /* "media=<ref>", "voice=<ref>", ... */
    int32_t            companion_count;   /* 0 = none */
    const char*        prompt_cache;  /* a prompt catalog file, read at load, extended at free; NULL = RAM only */
    int32_t            context_capacity;  /* tokens of context each state holds, at most JINFER_MAX_CONTEXT_LENGTH; 0 = the default
                                             (4096 or the maximum when smaller; an embedder or reranker takes the maximum),
                                             -1 = the maximum; above it is JINFER_EINVAL */
} jinfer_model_options;

JINFER_API void          jinfer_model_options_init(jinfer_model_options* o);
JINFER_API jinfer_model* jinfer_model_load(const char* ref, const jinfer_model_options* o);  /* hub ref or path; NULL = defaults */
JINFER_API void          jinfer_model_free(jinfer_model* m);

typedef enum {   /* every value spelled out: they are the ABI */
    JINFER_VOCAB_SIZE          = 1,
    JINFER_CONTEXT_CAPACITY    = 2,
    JINFER_EMBEDDING_DIMENSION = 3,
    JINFER_SAMPLE_RATE         = 4,
    JINFER_MAX_CONTEXT_LENGTH  = 5,     /* what the model was trained for; JINFER_CONTEXT_CAPACITY is what its states hold */
    JINFER_HAS_CHAT            = 100,
    JINFER_HAS_MEDIA           = 101,   /* images and audio in, through the media companion */
    JINFER_HAS_EMBEDDING       = 102,
    JINFER_HAS_RERANK          = 103,
    JINFER_HAS_SPEECH          = 104,
    JINFER_HAS_TRANSCRIPTION   = 105
} jinfer_property;

JINFER_API int32_t     jinfer_model_property(const jinfer_model* m, jinfer_property key);  /* 0 = not applicable; JINFER_EINVAL = unknown key */
JINFER_API const char* jinfer_model_info(const jinfer_model* m);                           /* JSON: architecture, quantization, defaults */

JINFER_API toknroll_tokenizer* jinfer_model_tokenizer(jinfer_model* m);   /* the model's tokenizer, see toknroll.h; yours to toknroll_free, it outlives the model; a reranker, speaker or transcriber has none */

/* --------------------------------------- state: one sequence of your own, the raw door */
typedef struct jinfer_state jinfer_state;

JINFER_API jinfer_state* jinfer_state_new(jinfer_model* m, int32_t context_capacity);  /* tokens of context it holds; 0 = the model's JINFER_CONTEXT_CAPACITY */
JINFER_API void          jinfer_state_free(jinfer_state* s);
JINFER_API int32_t       jinfer_state_reset(jinfer_state* s);
JINFER_API int32_t       jinfer_state_position(const jinfer_state* s);

JINFER_API int32_t jinfer_state_ingest(jinfer_state* s, const int32_t* tokens, int32_t count);  /* forward pass only; returns the new position, JINFER_EINVAL past the capacity */
JINFER_API int32_t jinfer_state_logits(const jinfer_state* s, float* out, int32_t capacity);     /* projects the last position; JINFER_VOCAB_SIZE */
JINFER_API int32_t jinfer_state_embedding(const jinfer_state* s, float* out, int32_t capacity);  /* pools what was ingested; JINFER_EMBEDDING_DIMENSION */

/* ------------------------------ stream: every generation, stepped like a statement */
typedef struct jinfer_stream jinfer_stream;

JINFER_API int32_t      jinfer_stream_step(jinfer_stream* s);                           /* blocks; JINFER_OK = chunk ready, JINFER_DONE, JINFER_ECANCELLED, or an error */
JINFER_API const char*  jinfer_stream_text(const jinfer_stream* s);                     /* the chunk's text, "" if none */
JINFER_API const float* jinfer_stream_samples(const jinfer_stream* s, int32_t* count);  /* the chunk's mono PCM at JINFER_SAMPLE_RATE, NULL if none */
JINFER_API const char*  jinfer_stream_result(const jinfer_stream* s);                   /* after JINFER_DONE or a cancellation: the result as JSON; cancelled, the text produced so far and finish_reason "cancelled" */
JINFER_API void         jinfer_stream_cancel(jinfer_stream* s);                         /* from any thread; the step in progress, or the next, returns JINFER_ECANCELLED */
JINFER_API void         jinfer_stream_free(jinfer_stream* s);                           /* cancels if still running */

JINFER_API jinfer_stream* jinfer_chat(jinfer_model* m, const char* request_json);        /* OpenAI chat.completions in; result is the response */
JINFER_API jinfer_stream* jinfer_speak(jinfer_model* m, const char* request_json);       /* {"input": text, "speed": 1.0}; chunks are audio */
JINFER_API jinfer_stream* jinfer_transcribe(jinfer_model* m, const float* pcm, int32_t count, int32_t sample_rate);  /* result: text and timed words */

/* ---------------------------------------------------------- retrieval, prompt cache */
JINFER_API int32_t jinfer_embed(jinfer_model* m, const char* const* texts, int32_t count, int32_t dimension,
                                float* out, int32_t capacity);   /* count x dimension row-major; 0 = native */
JINFER_API int32_t jinfer_rerank(jinfer_model* m, const char* query, const char* const* documents, int32_t count,
                                 const char* instruction, float* scores);   /* scores[count]; NULL = model default */
JINFER_API int32_t jinfer_prepare(jinfer_model* m, const char* request_json);  /* computes and caches this prefix now */

#ifdef __cplusplus
}
#endif
#endif
