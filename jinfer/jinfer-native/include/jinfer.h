/* jinfer.h - local AI inference for any language. C99, one header, no dependencies.
 *
 *   Errors     A function returns its result; a negative value is a JINFER_E* code, NULL for a
 *              pointer. jinfer_last_error has the message and the code, per thread, until the
 *              thread's next jinfer call (reading it changes nothing); after a success, "" and
 *              JINFER_OK. Every call sets it but jinfer_version, jinfer_model_options_init and
 *              jinfer_last_error.
 *   Buffers    (out, capacity) follows snprintf: at most capacity elements are written and the
 *              number needed is returned. Capacity 0 sizes, and a size known up front (count x
 *              dimension) is answered before any work. Element counts are int32_t; the sizes a
 *              model fixes come from jinfer_model_property. The toknroll_* functions write text as
 *              toknroll.h says.
 *   Text       UTF-8, NUL-terminated. A function that returns a string never returns NULL: "" stands
 *              for none, and for a failure.
 *   Ownership  What you pass is borrowed for the call. Everything returned is borrowed from the handle
 *              it came from: a model's strings live as long as the model, a state's or a stream's
 *              until the next call on that handle. Nothing is freed but handles, each by its own
 *              _free, which accepts NULL. A model stays alive while any state or stream made from it
 *              lives; _free drops your reference.
 *   Threads    Any thread may call, and a model may be shared. A state or a stream takes one call at
 *              a time: a call while another thread's is running returns JINFER_EBUSY.
 *              jinfer_stream_cancel is the exception, any thread may call it at any time; a _free
 *              must not run during another call on the same handle. The log callback runs on any
 *              thread, inside your own call included, and may call only jinfer_last_error,
 *              toknroll_last_error and the _free functions, on a handle no call is using (a stream
 *              still generating is in use). Once jinfer_log_set returns, the callback it replaced is
 *              not running and is never called again, so its user pointer may be freed.
 *   Fork       A child forked once the library has started cannot use it, the toknroll_* functions it
 *              carries included: its threads stayed in the parent, so calls fail there, JINFER_ERROR,
 *              and a _free does nothing. Start children with spawn or exec.
 *   Settings   The library is a Java runtime compiled ahead of time. jinfer_configure passes it
 *              options as a java command line would, before the first call, and the JINFER_OPTIONS
 *              environment variable adds to or overrides them; nothing requires either. Log lines go
 *              to jinfer_log_set's callback, and nowhere without one. The library writes to stderr
 *              only the runtime's name for an option it rejects, and a download's progress when
 *              stderr is a terminal (log lines otherwise). Once started the library stays loaded, as
 *              its threads run its code: an unload (dlclose, FreeLibrary) succeeds and leaves it in
 *              place.
 *   Defaults   A request field left out takes the model file's value, listed by jinfer_model_info,
 *              never the library's.
 *   Downloads  run through curl on the PATH: the library carries no TLS stack of its own.
 *              -Djinfer.offline=true never downloads; local files need nothing.
 *   Payloads   Requests are the OpenAI JSON shapes. jinfer_chat takes and returns exactly the jinfer
 *              server's /v1/chat/completions request and response, tools, response_format with a JSON
 *              schema and media as data URLs included; jinfer_speak takes the /v1/audio/speech fields
 *              that apply (input, speed). Text inside a request is JSON text: escape it. A
 *              schema-constrained reply cannot come back malformed: tokens outside the schema are
 *              never sampled.
 *   Stability  Before 1.0 any release may break this API. From 1.0 a PATCH changes nothing a caller
 *              relies on, a MINOR only adds, a MAJOR may break, and every release is checked.
 */
#ifndef JINFER_H
#define JINFER_H
#include <stddef.h>
#include <stdint.h>
#include "toknroll.h"   /* the tokenizer API: jinfer_model_tokenizer hands out a toknroll_tokenizer */
#ifdef __cplusplus
extern "C" {
#endif
#if defined(JINFER_STATIC)
#  define JINFER_API
#elif defined(_WIN32) && defined(JINFER_BUILD)
#  define JINFER_API __declspec(dllexport)
#elif defined(_WIN32)
#  define JINFER_API __declspec(dllimport)
#elif defined(__GNUC__)
#  define JINFER_API __attribute__((visibility("default")))
#else
#  define JINFER_API
#endif

#define JINFER_VERSION_MAJOR 0
#define JINFER_VERSION_MINOR 4
#define JINFER_VERSION_PATCH 0

#define JINFER_OK            0
#define JINFER_DONE          1    /* a stream has nothing more to produce */
#define JINFER_ERROR        (-1) /* see jinfer_last_error() */
#define JINFER_EINVAL       (-2)
#define JINFER_ENOTFOUND    (-3) /* model, file or companion */
#define JINFER_EUNSUPPORTED (-4) /* the model lacks this capability */
#define JINFER_ECANCELLED   (-5)
#define JINFER_EBUSY        (-6) /* a state or a stream in a call on another thread */

enum { JINFER_LOG_DEBUG = 0, JINFER_LOG_INFO = 1, JINFER_LOG_WARN = 2, JINFER_LOG_ERROR = 3 };   /* log levels */
typedef void (*jinfer_log_fn)(int32_t level, const char* message, void* user);

/* "0.4.0", the jinfer release the library was built from. */
JINFER_API const char* jinfer_version(void);
/* The message of the thread's last call; *code = its code unless code is NULL. "" and JINFER_OK after a
 * success. */
JINFER_API const char* jinfer_last_error(int32_t* code);
/* The callback the library's log lines go to, with user as its last argument; NULL = silent, the default.
 * May precede jinfer_configure. */
JINFER_API void        jinfer_log_set(jinfer_log_fn fn, void* user);
/* Runtime options, as on a java command line: "-Djinfer.threads=8" sizes the process's pool,
 * "-Djinfer.cache.sessions=16" keeps that many conversations warm (4), "-Djinfer.cache.budget=8g" bounds
 * the prompt cache's KV blocks (2g, 0 = off), "-Djinfer.offline=true" never downloads,
 * "-Djinfer.trace=true", "-Djam.*" and the other jinfer.* properties steer the engine,
 * "-XX:MaxHeapSize=4g" caps the heap. Once, before any other call into the library; a later call is
 * JINFER_EINVAL. toknroll_configure is the same call on the same runtime. An option the runtime rejects
 * fails this call, JINFER_ERROR, the runtime naming it on stderr (an unknown -XX: option is rejected, an
 * unknown -X one is ignored). JINFER_OPTIONS in the environment, the same options separated by
 * whitespace, is read when the runtime starts and applied after these, so it overrides them: the program
 * sets its options, whoever runs it has the last word. */
JINFER_API int32_t     jinfer_configure(const char* const* options, int32_t count);

/* ------------------------------------------------------------------ model */
/* A loaded model: the weights and configuration, plus the states that jinfer_chat, jinfer_embed and
 * jinfer_rerank run in, each holding up to context_capacity tokens of context. */
typedef struct jinfer_model jinfer_model;

/* Initialized by jinfer_model_options_init, which sets struct_size and leaves every other field zero,
 * its default; a struct_size smaller than this struct is JINFER_EINVAL. */
typedef struct {
    size_t             struct_size;   /* set by jinfer_model_options_init */
    const char* const* companions;    /* "media=<ref>", "voice=<ref>", ... */
    int32_t            companion_count;   /* 0 = none */
    /* a prompt catalog file, read at load, extended at free; NULL = RAM only */
    const char*        prompt_cache;
    /* tokens of context each state holds, at most JINFER_MAX_CONTEXT_LENGTH; 0 = the default (4096 or the
     * maximum when smaller; an embedder or reranker takes the maximum), -1 = the maximum; above it is
     * JINFER_EINVAL */
    int32_t            context_capacity;
} jinfer_model_options;

JINFER_API void          jinfer_model_options_init(jinfer_model_options* o);
/* A model from a hub reference or a path; options NULL = all defaults. */
JINFER_API jinfer_model* jinfer_model_load(const char* ref, const jinfer_model_options* o);
JINFER_API void          jinfer_model_free(jinfer_model* m);

enum {   /* jinfer_model_property's keys; every value spelled out: they are the ABI */
    JINFER_VOCAB_SIZE          = 1,
    JINFER_CONTEXT_CAPACITY    = 2,
    JINFER_EMBEDDING_DIMENSION = 3,
    JINFER_SAMPLE_RATE         = 4,
    JINFER_MAX_CONTEXT_LENGTH  = 5,     /* what the model was trained for; CONTEXT_CAPACITY what its states hold */
    JINFER_HAS_CHAT            = 100,
    JINFER_HAS_MEDIA           = 101,   /* images and audio in, through the media companion */
    JINFER_HAS_EMBEDDING       = 102,
    JINFER_HAS_RERANK          = 103,
    JINFER_HAS_SPEECH          = 104,
    JINFER_HAS_TRANSCRIPTION   = 105
};

/* A JINFER_* key's value; 0 = not applicable, JINFER_EINVAL = an unknown key. */
JINFER_API int32_t     jinfer_model_property(const jinfer_model* m, int32_t key);
/* JSON: the architecture, the quantization, the defaults; "" on error. */
JINFER_API const char* jinfer_model_info(const jinfer_model* m);

/* The model's tokenizer, see toknroll.h: yours to toknroll_tokenizer_free, and it outlives the model. A
 * reranker, speaker or transcriber has none. */
JINFER_API toknroll_tokenizer* jinfer_model_tokenizer(const jinfer_model* m);

/* --------------------------------------- state: one sequence of your own, the raw door */
typedef struct jinfer_state jinfer_state;

/* Holds context_capacity tokens of context; 0 = the model's JINFER_CONTEXT_CAPACITY. */
JINFER_API jinfer_state* jinfer_state_new(jinfer_model* m, int32_t context_capacity);
JINFER_API void          jinfer_state_free(jinfer_state* s);
JINFER_API int32_t       jinfer_state_reset(jinfer_state* s);
JINFER_API int32_t       jinfer_state_position(const jinfer_state* s);

/* A forward pass only; returns the new position, JINFER_EINVAL past the capacity. */
JINFER_API int32_t jinfer_state_ingest(jinfer_state* s, const int32_t* tokens, int32_t count);
/* The last position projected, JINFER_VOCAB_SIZE floats. */
JINFER_API int32_t jinfer_state_logits(const jinfer_state* s, float* out, int32_t capacity);
/* What was ingested pooled, JINFER_EMBEDDING_DIMENSION floats. */
JINFER_API int32_t jinfer_state_embedding(const jinfer_state* s, float* out, int32_t capacity);

/* ------------------------------ stream: every generation, stepped like a statement */
typedef struct jinfer_stream jinfer_stream;

/* Blocks; JINFER_OK = a chunk is ready, JINFER_DONE, JINFER_ECANCELLED, or an error. */
JINFER_API int32_t      jinfer_stream_step(jinfer_stream* s);
/* The chunk's text; "" if none, or on error. */
JINFER_API const char*  jinfer_stream_text(const jinfer_stream* s);
/* The chunk's mono PCM at JINFER_SAMPLE_RATE, *count samples; NULL if none. */
JINFER_API const float* jinfer_stream_samples(const jinfer_stream* s, int32_t* count);
/* After JINFER_DONE or a cancellation, the result as JSON; cancelled, the text produced so far and
 * finish_reason "cancelled". "" before then, or on error. */
JINFER_API const char*  jinfer_stream_result(const jinfer_stream* s);
/* From any thread: the step in progress, or the next, returns JINFER_ECANCELLED. */
JINFER_API void         jinfer_stream_cancel(jinfer_stream* s);
JINFER_API void         jinfer_stream_free(jinfer_stream* s);                           /* cancels if still running */

/* An OpenAI chat.completions request in; the result is the response. */
JINFER_API jinfer_stream* jinfer_chat(jinfer_model* m, const char* request_json);
/* {"input": text, "speed": 1.0} in; the chunks are audio. */
JINFER_API jinfer_stream* jinfer_speak(jinfer_model* m, const char* request_json);
/* The result is the text and its timed words. */
JINFER_API jinfer_stream* jinfer_transcribe(jinfer_model* m, const float* pcm, int32_t count, int32_t sample_rate);

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
