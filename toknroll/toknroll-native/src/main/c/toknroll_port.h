/* What the shims need from the platform, and nothing more: run a function once, declare a thread-local
 * variable, keep one value per thread with a destructor at thread exit, duplicate a string, read and write
 * a flag across threads, lock, wait and wake (the batch workers), start a detached thread with a given
 * stack, and check a printf format. POSIX everywhere but Windows, where the same come from the Win32 API
 * (fiber-local storage is the thread-local storage that has destructors). */
#ifndef TOKNROLL_PORT_H
#define TOKNROLL_PORT_H
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
typedef INIT_ONCE port_once_t;
#define PORT_ONCE_INIT INIT_ONCE_STATIC_INIT
typedef DWORD port_tls_t;
/* the function travels by address: ISO C converts no function pointer to void* */
static inline BOOL CALLBACK port_call_once(PINIT_ONCE once, PVOID fn, PVOID* context) {
    (void) once;
    (void) context;
    (*(void (**)(void)) fn)();
    return TRUE;
}
static inline void port_once(port_once_t* once, void (*fn)(void)) { InitOnceExecuteOnce(once, port_call_once, &fn, NULL); }
static inline int port_tls_create(port_tls_t* key, void (*destructor)(void*)) { return (*key = FlsAlloc((PFLS_CALLBACK_FUNCTION) destructor)) == FLS_OUT_OF_INDEXES; }
#define port_tls_get(key) FlsGetValue(key)
static inline int port_tls_set(port_tls_t key, void* value) { return FlsSetValue(key, value) ? 0 : -1; }
#define port_tls_delete(key) FlsFree(key)
#define port_strdup _strdup
#define PORT_THREAD_LOCAL __declspec(thread)
#define port_load(flag) InterlockedCompareExchange((flag), 0, 0)   /* flag: a long* */
#define port_store(flag, value) InterlockedExchange((flag), (value))
/* an int32_t counter: its value, the value before adding, and 0 replaced by value (else unchanged) */
#define port_load32(p) ((int32_t) InterlockedCompareExchange((volatile long*) (p), 0, 0))
#define port_fetch_add32(p, v) ((int32_t) InterlockedExchangeAdd((volatile long*) (p), (v)))
#define port_set_if_zero32(p, v) ((void) InterlockedCompareExchange((volatile long*) (p), (v), 0))
typedef SRWLOCK port_mutex_t;
#define PORT_MUTEX_INIT SRWLOCK_INIT
#define port_mutex_lock AcquireSRWLockExclusive
#define port_mutex_unlock ReleaseSRWLockExclusive
typedef CONDITION_VARIABLE port_cond_t;
#define PORT_COND_INIT CONDITION_VARIABLE_INIT
static inline void port_cond_wait(port_cond_t* cond, port_mutex_t* mutex) { SleepConditionVariableSRW(cond, mutex, INFINITE, 0); }
#define port_cond_broadcast WakeAllConditionVariable
/* a thread's function: PORT_THREAD(name, arg) { ... return PORT_THREAD_END; } */
#define PORT_THREAD(name, arg) static DWORD WINAPI name(LPVOID arg)
#define PORT_THREAD_END 0
static inline int port_thread_start(LPTHREAD_START_ROUTINE fn, void* arg, size_t stack) {
    HANDLE thread = CreateThread(NULL, stack, fn, arg, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!thread) return -1;
    CloseHandle(thread);   /* detached: it lives as long as the process */
    return 0;
}
#define port_thread_name(name) ((void) (name))   /* ponytail: unnamed on Windows; SetThreadDescription if a debugger needs it */
#else
#define PORT_THREAD_LOCAL _Thread_local
#include <pthread.h>
typedef pthread_once_t port_once_t;
#define PORT_ONCE_INIT PTHREAD_ONCE_INIT
typedef pthread_key_t port_tls_t;
#define port_once pthread_once
#define port_tls_create pthread_key_create
#define port_tls_get pthread_getspecific
#define port_tls_set pthread_setspecific
#define port_tls_delete pthread_key_delete
#define port_strdup strdup
#define port_load(flag) __atomic_load_n((flag), __ATOMIC_SEQ_CST)
#define port_store(flag, value) __atomic_store_n((flag), (value), __ATOMIC_SEQ_CST)
#define port_load32(p) __atomic_load_n((p), __ATOMIC_SEQ_CST)
#define port_fetch_add32(p, v) __atomic_fetch_add((p), (v), __ATOMIC_SEQ_CST)
#define port_set_if_zero32(p, v) ((void) __sync_bool_compare_and_swap((p), 0, (v)))
typedef pthread_mutex_t port_mutex_t;
#define PORT_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define port_mutex_lock pthread_mutex_lock
#define port_mutex_unlock pthread_mutex_unlock
typedef pthread_cond_t port_cond_t;
#define PORT_COND_INIT PTHREAD_COND_INITIALIZER
#define port_cond_wait pthread_cond_wait
#define port_cond_broadcast pthread_cond_broadcast
#define PORT_THREAD(name, arg) static void* name(void* arg)
#define PORT_THREAD_END NULL
static inline int port_thread_start(void* (*fn)(void*), void* arg, size_t stack) {
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) return -1;
    pthread_t thread;
    int rc = pthread_attr_setstacksize(&attr, stack) != 0 || pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED) != 0
             || pthread_create(&thread, &attr, fn, arg) != 0;
    pthread_attr_destroy(&attr);
    return rc ? -1 : 0;
}
/* the calling thread's name, as debuggers and /proc show it: at most 15 bytes on Linux */
#if defined(__APPLE__)
#define port_thread_name(name) ((void) pthread_setname_np(name))
#else
#define port_thread_name(name) ((void) pthread_setname_np(pthread_self(), (name)))
#endif
#endif

#if defined(__GNUC__)
#define PORT_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#else
#define PORT_PRINTF(fmt, args)
#endif
#endif
