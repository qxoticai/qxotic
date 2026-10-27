/* The C face of libjinfer: forwards each header function to the jinfer__* entry point the image
 * exports, which takes the isolate thread first. The isolate is toknroll_isolate.c's, shared with the
 * toknroll_* symbols this library carries; jinfer_configure and toknroll_configure feed the same one.
 * Nothing here knows what a model is. */
#include <stdint.h>
#include <string.h>
#include "jinfer.h"
#include "toknroll_isolate.h"   /* toknroll-native/src/main/c */
#include "toknroll_port.h"

void jinfer__error_set(graal_thread*, int32_t, const char*);
void jinfer__log_set(graal_thread*, jinfer_log_fn, void*);

/* a log callback installed before the isolate exists is applied on the first call that has one */
static jinfer_log_fn pending_log;
static void* pending_payload;
static port_once_t log_once = PORT_ONCE_INIT;
static void apply_log(void) { jinfer__log_set(toknroll__thread(), pending_log, pending_payload); }

static graal_thread* thread(void) {
    graal_thread* t = toknroll__thread();
    if (t && pending_log) port_once(&log_once, apply_log);
    return t;
}

typedef graal_thread* T;
#define THREAD T t = thread()
#define STR(x) #x
#define XSTR(x) STR(x)

const char* jinfer__last_error(T, int32_t*);
jinfer_model* jinfer__model_load(T, const char*, const jinfer_model_options*);
void jinfer__model_free(T, jinfer_model*);
int32_t jinfer__model_property(T, const jinfer_model*, int32_t);
const char* jinfer__model_info(T, const jinfer_model*);
toknroll_tokenizer* jinfer__model_tokenizer(T, jinfer_model*);
jinfer_state* jinfer__state_new(T, jinfer_model*, int32_t);
void jinfer__state_free(T, jinfer_state*);
int32_t jinfer__state_reset(T, jinfer_state*);
int32_t jinfer__state_position(T, const jinfer_state*);
int32_t jinfer__state_ingest(T, jinfer_state*, const int32_t*, int32_t);
int32_t jinfer__state_logits(T, const jinfer_state*, float*, int32_t);
int32_t jinfer__state_embedding(T, const jinfer_state*, float*, int32_t);
int32_t jinfer__stream_step(T, jinfer_stream*);
const char* jinfer__stream_text(T, const jinfer_stream*);
const float* jinfer__stream_samples(T, const jinfer_stream*, int32_t*);
const char* jinfer__stream_result(T, const jinfer_stream*);
void jinfer__stream_cancel(T, jinfer_stream*);
void jinfer__stream_free(T, jinfer_stream*);
jinfer_stream* jinfer__chat(T, jinfer_model*, const char*);
jinfer_stream* jinfer__speak(T, jinfer_model*, const char*);
jinfer_stream* jinfer__transcribe(T, jinfer_model*, const float*, int32_t, int32_t);
int32_t jinfer__embed(T, jinfer_model*, const char* const*, int32_t, int32_t, float*, int32_t);
int32_t jinfer__rerank(T, jinfer_model*, const char*, const char* const*, int32_t, const char*, float*);
int32_t jinfer__prepare(T, jinfer_model*, const char*);

const char* jinfer_version(void) { return XSTR(JINFER_VERSION_MAJOR) "." XSTR(JINFER_VERSION_MINOR) "." XSTR(JINFER_VERSION_PATCH); }
/* never create the isolate just to report on it, nor apply a pending log callback: that is a call, and
 * a call clears the error being read */
const char* jinfer_last_error(int32_t* code) { T t = toknroll__created() ? toknroll__thread() : NULL; return t ? jinfer__last_error(t, code) : toknroll__early_error(code); }
void jinfer_log_set(jinfer_log_fn fn, void* payload) {
    if (!toknroll__created()) { pending_log = fn; pending_payload = payload; return; }
    THREAD;
    if (t) jinfer__log_set(t, fn, payload);
}
int32_t jinfer_configure(const char* const* options, int32_t count) { return toknroll__configure(options, count, "jinfer_configure", jinfer__error_set); }

void jinfer_model_options_init(jinfer_model_options* o) { memset(o, 0, sizeof *o); o->struct_size = sizeof *o; }
jinfer_model* jinfer_model_load(const char* ref, const jinfer_model_options* o) { THREAD; return t ? jinfer__model_load(t, ref, o) : NULL; }
void jinfer_model_free(jinfer_model* m) { if (!m) return; THREAD; if (t) jinfer__model_free(t, m); }   /* NULL never starts the runtime, here and below */
int32_t jinfer_model_property(const jinfer_model* m, jinfer_property key) { THREAD; return t ? jinfer__model_property(t, m, (int32_t) key) : JINFER_ERROR; }
const char* jinfer_model_info(const jinfer_model* m) { THREAD; return t ? jinfer__model_info(t, m) : NULL; }
toknroll_tokenizer* jinfer_model_tokenizer(jinfer_model* m) { THREAD; return t ? jinfer__model_tokenizer(t, m) : NULL; }

jinfer_state* jinfer_state_new(jinfer_model* m, int32_t capacity) { THREAD; return t ? jinfer__state_new(t, m, capacity) : NULL; }
void jinfer_state_free(jinfer_state* s) { if (!s) return; THREAD; if (t) jinfer__state_free(t, s); }
int32_t jinfer_state_reset(jinfer_state* s) { THREAD; return t ? jinfer__state_reset(t, s) : JINFER_ERROR; }
int32_t jinfer_state_position(const jinfer_state* s) { THREAD; return t ? jinfer__state_position(t, s) : JINFER_ERROR; }
int32_t jinfer_state_ingest(jinfer_state* s, const int32_t* tokens, int32_t count) { THREAD; return t ? jinfer__state_ingest(t, s, tokens, count) : JINFER_ERROR; }
int32_t jinfer_state_logits(const jinfer_state* s, float* out, int32_t cap) { THREAD; return t ? jinfer__state_logits(t, s, out, cap) : JINFER_ERROR; }
int32_t jinfer_state_embedding(const jinfer_state* s, float* out, int32_t cap) { THREAD; return t ? jinfer__state_embedding(t, s, out, cap) : JINFER_ERROR; }

int32_t jinfer_stream_step(jinfer_stream* s) { THREAD; return t ? jinfer__stream_step(t, s) : JINFER_ERROR; }
const char* jinfer_stream_text(const jinfer_stream* s) { THREAD; return t ? jinfer__stream_text(t, s) : ""; }
const float* jinfer_stream_samples(const jinfer_stream* s, int32_t* count) { THREAD; if (!t) { if (count) *count = 0; return NULL; } return jinfer__stream_samples(t, s, count); }
const char* jinfer_stream_result(const jinfer_stream* s) { THREAD; return t ? jinfer__stream_result(t, s) : NULL; }
void jinfer_stream_cancel(jinfer_stream* s) { if (!s) return; THREAD; if (t) jinfer__stream_cancel(t, s); }
void jinfer_stream_free(jinfer_stream* s) { if (!s) return; THREAD; if (t) jinfer__stream_free(t, s); }

jinfer_stream* jinfer_chat(jinfer_model* m, const char* request_json) { THREAD; return t ? jinfer__chat(t, m, request_json) : NULL; }
jinfer_stream* jinfer_speak(jinfer_model* m, const char* request_json) { THREAD; return t ? jinfer__speak(t, m, request_json) : NULL; }
jinfer_stream* jinfer_transcribe(jinfer_model* m, const float* pcm, int32_t count, int32_t rate) { THREAD; return t ? jinfer__transcribe(t, m, pcm, count, rate) : NULL; }

int32_t jinfer_embed(jinfer_model* m, const char* const* texts, int32_t count, int32_t dim, float* out, int32_t cap) { THREAD; return t ? jinfer__embed(t, m, texts, count, dim, out, cap) : JINFER_ERROR; }
int32_t jinfer_rerank(jinfer_model* m, const char* query, const char* const* docs, int32_t count, const char* instruction, float* scores) { THREAD; return t ? jinfer__rerank(t, m, query, docs, count, instruction, scores) : JINFER_ERROR; }
int32_t jinfer_prepare(jinfer_model* m, const char* request_json) { THREAD; return t ? jinfer__prepare(t, m, request_json) : JINFER_ERROR; }
