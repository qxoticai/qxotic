/* A small persistent worker pool for contexts without a host executor: the C global context, the
 * tests and the benches. Workers spin on the job counter for a bounded number of pauses, then park
 * on a condvar until the next job; the submitting thread is participant 0. A JVM host never uses
 * it - the Java context supplies parallel_for and this file's threads are never created there. */
#include "jam_internal.h"

#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>

#if defined(__x86_64__) || defined(_M_X64)
#  define JAM_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#  define JAM_PAUSE() __asm__ __volatile__("yield")
#else
#  define JAM_PAUSE() ((void) 0)
#endif

#define JAM_SPIN_PAUSES 16384   /* ~100 us of spinning before a worker parks */

typedef struct { jam_pool* pool; int idx; } jam_worker;

struct jam_pool {
    int             nworkers;     /* worker THREADS; participants = nworkers + 1 (the submitter) */
    pthread_t*      threads;
    jam_worker*     wargs;
    pthread_mutex_t mtx;
    pthread_cond_t  cv;           /* parked workers wait here for a job */
    jam_task_fn     fn;
    void*           arg;
    int             n;
    _Atomic uint32_t seq;         /* (generation << 16) | participants: one read = one consistent job */
    _Atomic int     remaining;    /* workers still running the current job */
    _Atomic int     parked;
    _Atomic int     stop;
};

/* Participant `idx` of `total` takes a balanced slice of [0, n). */
static void run_range(jam_task_fn fn, void* arg, int n, int idx, int total) {
    int chunk = n / total, rem = n % total;
    int begin = idx * chunk + (idx < rem ? idx : rem);
    int end   = begin + chunk + (idx < rem ? 1 : 0);
    if (begin < end) fn(arg, begin, end, idx);
}

static void* worker_main(void* p) {
    jam_worker* w = (jam_worker*) p;
    jam_pool* pool = w->pool;
    uint32_t my_seq = 0;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&pool->seq, memory_order_acquire) == my_seq
               && !atomic_load_explicit(&pool->stop, memory_order_relaxed)) {
            if (++spins < JAM_SPIN_PAUSES) { JAM_PAUSE(); continue; }
            pthread_mutex_lock(&pool->mtx);   /* idle too long: park, re-checking seq under the lock */
            atomic_fetch_add(&pool->parked, 1);
            while (atomic_load(&pool->seq) == my_seq && !atomic_load(&pool->stop))
                pthread_cond_wait(&pool->cv, &pool->mtx);
            atomic_fetch_sub(&pool->parked, 1);
            pthread_mutex_unlock(&pool->mtx);
            spins = 0;
        }
        if (atomic_load(&pool->stop)) break;
        /* A participant's job cannot complete without it, so fn/arg/n are pinned while it reads
         * them; a worker at or above the fan width reads nothing and just adopts the generation. */
        uint32_t job = atomic_load_explicit(&pool->seq, memory_order_acquire);
        int part = (int) (job & 0xFFFF);
        if (w->idx < part) {
            run_range(pool->fn, pool->arg, pool->n, w->idx, part);
            atomic_fetch_sub_explicit(&pool->remaining, 1, memory_order_acq_rel);
        }
        my_seq = job;
    }
    return NULL;
}

jam_pool* jam_pool_create(int nthreads) {
    if (nthreads < 1) nthreads = 1;
    jam_pool* pool = (jam_pool*) calloc(1, sizeof *pool);
    if (!pool) return NULL;
    pool->nworkers = nthreads - 1;
    pthread_mutex_init(&pool->mtx, NULL);
    pthread_cond_init(&pool->cv, NULL);
    if (pool->nworkers > 0) {
        pool->threads = (pthread_t*) calloc(pool->nworkers, sizeof(pthread_t));
        pool->wargs   = (jam_worker*) calloc(pool->nworkers, sizeof(jam_worker));
        for (int i = 0; i < pool->nworkers; ++i) {
            pool->wargs[i].pool = pool;
            pool->wargs[i].idx  = i + 1;   /* participants 1..nworkers; the submitter is 0 */
            pthread_create(&pool->threads[i], NULL, worker_main, &pool->wargs[i]);
        }
    }
    return pool;
}

void jam_pool_parallel_for(jam_pool* pool, int n, jam_task_fn fn, void* arg) {
    if (!pool || pool->nworkers == 0 || n <= 1) { fn(arg, 0, n, 0); return; }
    pool->fn = fn; pool->arg = arg; pool->n = n;
    int part = pool->nworkers + 1;
    if (part > n) part = n;               /* never fan wider than the work units */
    uint32_t next = ((atomic_load_explicit(&pool->seq, memory_order_relaxed) >> 16) + 1) << 16
                    | (uint32_t) part;
    atomic_store_explicit(&pool->remaining, part - 1, memory_order_relaxed);
    atomic_store_explicit(&pool->seq, next, memory_order_seq_cst);   /* publish (also orders vs parked) */
    if (atomic_load_explicit(&pool->parked, memory_order_seq_cst) > 0) {
        pthread_mutex_lock(&pool->mtx);
        pthread_cond_broadcast(&pool->cv);
        pthread_mutex_unlock(&pool->mtx);
    }
    run_range(fn, arg, n, 0, part);                                  /* the submitter is participant 0 */
    while (atomic_load_explicit(&pool->remaining, memory_order_acquire) > 0) JAM_PAUSE();
}

void jam_pool_destroy(jam_pool* pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mtx);
    atomic_store(&pool->stop, 1);
    pthread_cond_broadcast(&pool->cv);
    pthread_mutex_unlock(&pool->mtx);
    for (int i = 0; i < pool->nworkers; ++i) pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->mtx);
    pthread_cond_destroy(&pool->cv);
    free(pool->threads);
    free(pool->wargs);
    free(pool);
}
