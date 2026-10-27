/* What the shims need from toknroll_isolate.c: the calling thread's isolate thread, the state before there
 * is a runtime, and configure. Private to the libraries: nothing declared here is exported. */
#ifndef TOKNROLL_ISOLATE_H
#define TOKNROLL_ISOLATE_H
#include <stdint.h>

/* GraalVM's graal_isolatethread_t, opaque: the first argument of every entry point the image exports.
 * Named here, not by GraalVM's header, which exists only once the image is built. */
typedef struct graal_thread graal_thread;

graal_thread* toknroll__thread(void);   /* attached on first use; NULL when there is no runtime */
graal_thread* toknroll__enter(void);    /* toknroll__thread inside the fork gate; always followed by leave */
void toknroll__leave(void);
int toknroll__created(void);            /* the runtime has started, or failed to: configure is too late */
const char* toknroll__early_error(int32_t* code);   /* the last failure no runtime could hold */
/* configure for the public function `who`; error_set reports through that API's own last error */
int32_t toknroll__configure(const char* const* options, int32_t count, const char* who,
                            void (*error_set)(graal_thread*, int32_t, const char*));
#endif
