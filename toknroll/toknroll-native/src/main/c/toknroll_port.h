/* What the shims need from the platform, and nothing more: run a function once, keep one value per thread
 * with a destructor at thread exit, duplicate a string, read and write a flag across threads, and check a
 * printf format. POSIX everywhere but Windows, where the same come from the Win32 API (fiber-local storage
 * is the thread-local storage that has destructors). */
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
#define port_tls_set(key, value) FlsSetValue(key, value)
#define port_strdup _strdup
#define port_load(flag) InterlockedCompareExchange((flag), 0, 0)   /* flag: a long* */
#define port_store(flag, value) InterlockedExchange((flag), (value))
#else
#include <pthread.h>
typedef pthread_once_t port_once_t;
#define PORT_ONCE_INIT PTHREAD_ONCE_INIT
typedef pthread_key_t port_tls_t;
#define port_once pthread_once
#define port_tls_create pthread_key_create
#define port_tls_get pthread_getspecific
#define port_tls_set pthread_setspecific
#define port_strdup strdup
#define port_load(flag) __atomic_load_n((flag), __ATOMIC_SEQ_CST)
#define port_store(flag, value) __atomic_store_n((flag), (value), __ATOMIC_SEQ_CST)
#endif

#if defined(__GNUC__)
#define PORT_PRINTF(fmt, args) __attribute__((format(printf, fmt, args)))
#else
#define PORT_PRINTF(fmt, args)
#endif
#endif
