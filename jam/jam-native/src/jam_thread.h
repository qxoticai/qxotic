/* The threads jam's pool needs, from pthreads or from Win32: a thread to create and join, a mutex,
 * a condition variable. One name each, so the pool reads the same on every
 * platform; the MSVC runtime has no pthreads and MinGW's would travel inside a DLL. */
#ifndef JAM_THREAD_H
#define JAM_THREAD_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include <stdint.h>
#include <stdlib.h>

typedef HANDLE jam_thread_t;
typedef SRWLOCK jam_mutex_t;
typedef CONDITION_VARIABLE jam_cond_t;

typedef struct { void* (*fn)(void*); void* arg; } jam_thread_start;
static inline unsigned __stdcall jam_thread_main(void* p) {
    jam_thread_start s = *(jam_thread_start*) p;
    free(p);
    s.fn(s.arg);
    return 0;
}
static inline int jam_thread_create(jam_thread_t* t, void* (*fn)(void*), void* arg) {
    jam_thread_start* s = (jam_thread_start*) malloc(sizeof *s);
    if (!s) return -1;
    s->fn = fn; s->arg = arg;
    uintptr_t h = _beginthreadex(NULL, 0, jam_thread_main, s, 0, NULL);
    if (!h) { free(s); return -1; }
    *t = (HANDLE) h;
    return 0;
}
static inline void jam_thread_join(jam_thread_t t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
static inline int  jam_mutex_init(jam_mutex_t* m)   { InitializeSRWLock(m); return 0; }
static inline void jam_mutex_destroy(jam_mutex_t* m){ (void) m; }
static inline void jam_mutex_lock(jam_mutex_t* m)   { AcquireSRWLockExclusive(m); }
static inline void jam_mutex_unlock(jam_mutex_t* m) { ReleaseSRWLockExclusive(m); }
static inline int  jam_cond_init(jam_cond_t* c)     { InitializeConditionVariable(c); return 0; }
static inline void jam_cond_destroy(jam_cond_t* c)  { (void) c; }
static inline void jam_cond_wait(jam_cond_t* c, jam_mutex_t* m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static inline void jam_cond_broadcast(jam_cond_t* c){ WakeAllConditionVariable(c); }

#else
#include <pthread.h>

typedef pthread_t jam_thread_t;
typedef pthread_mutex_t jam_mutex_t;
typedef pthread_cond_t jam_cond_t;

static inline int  jam_thread_create(jam_thread_t* t, void* (*fn)(void*), void* arg) { return pthread_create(t, NULL, fn, arg); }
static inline void jam_thread_join(jam_thread_t t)  { pthread_join(t, NULL); }
static inline int  jam_mutex_init(jam_mutex_t* m)   { return pthread_mutex_init(m, NULL); }
static inline void jam_mutex_destroy(jam_mutex_t* m){ pthread_mutex_destroy(m); }
static inline void jam_mutex_lock(jam_mutex_t* m)   { pthread_mutex_lock(m); }
static inline void jam_mutex_unlock(jam_mutex_t* m) { pthread_mutex_unlock(m); }
static inline int  jam_cond_init(jam_cond_t* c)     { return pthread_cond_init(c, NULL); }
static inline void jam_cond_destroy(jam_cond_t* c)  { pthread_cond_destroy(c); }
static inline void jam_cond_wait(jam_cond_t* c, jam_mutex_t* m) { pthread_cond_wait(c, m); }
static inline void jam_cond_broadcast(jam_cond_t* c){ pthread_cond_broadcast(c); }
#endif
#endif
