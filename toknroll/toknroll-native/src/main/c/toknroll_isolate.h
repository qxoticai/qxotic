/* What the shims need from toknroll_isolate.c: the calling thread's isolate thread, the state before there
 * is a runtime, configure and log_set; and STR, which all three use. Private to the libraries: nothing
 * declared here is exported. Workers.java's batch, toknroll__batch, is the image's to call. */
#ifndef TOKNROLL_ISOLATE_H
#define TOKNROLL_ISOLATE_H
#include <stdint.h>

/* a macro's value as a string literal: STR(TOKNROLL_PROGRAM) is "toknroll" */
#define STR_(x) #x
#define STR(x) STR_(x)

/* GraalVM's graal_isolatethread_t, opaque: the first argument of every entry point the image exports.
 * Named here, not by GraalVM's header, which exists only once the image is built. */
typedef struct graal_thread graal_thread;

graal_thread* toknroll__thread(void);   /* attached on first use; NULL when there is no runtime */
graal_thread* toknroll__enter(void);    /* toknroll__thread inside the fork gate; always followed by leave */
void toknroll__leave(void);
int toknroll__created(void);            /* the runtime has started, or failed to: configure is too late */
/* toknroll_log_set and jinfer_log_set for the public function `who`: kept for the runtime's start */
typedef void (*toknroll__log_fn)(int32_t level, const char* message, void* user_data);
int32_t toknroll__log(toknroll__log_fn fn, void* user_data, const char* who,
                      void (*error_set)(graal_thread*, int32_t, const char*));
/* toknroll_last_error and jinfer_last_error: the thread's own early failure, else the runtime's record */
const char* toknroll__last_error_of(int32_t* code, const char* (*runtime)(graal_thread*, int32_t*));
/* Each API's message storage, 1024 UTF-8 bytes plus NUL. allocate != 0 creates it on first error,
 * or returns NULL on failure; allocate == 0 only reads, returning static fallback text if absent.
 * Buffers are freed by native TLS at thread exit, independently of the runtime's attachment. */
const char* toknroll__error_buffer(int32_t allocate);
const char* jinfer__error_buffer(int32_t allocate);
/* configure for the public function `who`; error_set reports through that API's own last error */
int32_t toknroll__configure(const char* const* options, int32_t count, const char* who,
                            void (*error_set)(graal_thread*, int32_t, const char*));
/* The quarantine of tokenizers an escape may have left half-updated: the thread's escaped flag, set to
 * value and returned as it was; whether a tokenizer is refused, the thread's error then saying so; refuse
 * one; free one. */
int toknroll__escape_swap(int value);
int toknroll__quarantined(const void* tokenizer);
void toknroll__quarantine(const void* tokenizer);
void toknroll__release(const void* tokenizer);
#endif
