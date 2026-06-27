/* A small persistent worker pool. Two barrier implementations, selectable per pool via JAM_POOL:
 *  - CONDVAR (default): workers park on a condvar between jobs — yields the CPU, ~µs wakeup.
 *  - SPIN (JAM_POOL=spin): workers busy-wait on the atomic job counter (~100 ns wakeup), falling back
 *    to parking after JAM_SPIN pauses so they don't burn cores when idle (e.g. during jinfer's Java
 *    decode). The submitter publishes via an atomic store and spins on `remaining` for the join. Wins
 *    when fan-outs come back-to-back (two-phase matmul, MoE small matmuls) — workers stay hot so the
 *    gap between fan-outs collapses.
 * The submitting thread participates as worker 0, so `nthreads` participants share each parallel_for. */
#include "jam_internal.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

#if defined(__x86_64__) || defined(_M_X64)
#  define JAM_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#  define JAM_PAUSE() __asm__ __volatile__("yield")
#else
#  define JAM_PAUSE() ((void) 0)
#endif

typedef struct { jam_pool* pool; int idx; } jam_worker;

struct jam_pool {
    int             nworkers;     /* worker THREADS; total participants = nworkers + 1 (incl. submitter) */
    pthread_t*      threads;
    jam_worker*     wargs;
    pthread_mutex_t mtx;
    pthread_cond_t  cv_work;      /* park fallback: workers wait for a job */
    pthread_cond_t  cv_done;      /* condvar mode: submitter waits for completion */
    jam_task_fn     fn;
    void*           arg;
    int             n;
    _Atomic int     seq;          /* job counter; workers compare to detect new work */
    _Atomic int     remaining;    /* workers still running the current job */
    _Atomic int     parked;       /* workers currently parked (spin mode: gate the wakeup broadcast) */
    _Atomic int     stop;
    int             spin;         /* 0 = condvar, 1 = spin-then-park */
    int             spin_budget;  /* pauses before a spinning worker parks */
};

/* Participant `idx` of `total` takes a balanced slice of [0, n). */
static void run_range(jam_task_fn fn, void* arg, int n, int idx, int total, int tid) {
    int chunk = n / total, rem = n % total;
    int begin = idx * chunk + (idx < rem ? idx : rem);
    int end   = begin + chunk + (idx < rem ? 1 : 0);
    if (begin < end) fn(arg, begin, end, tid);
}

static void* worker_main(void* p) {
    jam_worker* w   = (jam_worker*) p;
    jam_pool*   pool = w->pool;
    int my_seq = 0;
    for (;;) {
        if (pool->spin) {
            int spins = 0;
            while (atomic_load_explicit(&pool->seq, memory_order_acquire) == my_seq
                   && !atomic_load_explicit(&pool->stop, memory_order_relaxed)) {
                if (++spins < pool->spin_budget) { JAM_PAUSE(); continue; }
                /* idle too long -> park; re-check seq UNDER the lock (seq_cst vs the publish) */
                pthread_mutex_lock(&pool->mtx);
                atomic_fetch_add(&pool->parked, 1);
                while (atomic_load(&pool->seq) == my_seq && !atomic_load(&pool->stop))
                    pthread_cond_wait(&pool->cv_work, &pool->mtx);
                atomic_fetch_sub(&pool->parked, 1);
                pthread_mutex_unlock(&pool->mtx);
                spins = 0;
            }
        } else {
            pthread_mutex_lock(&pool->mtx);
            while (atomic_load(&pool->seq) == my_seq && !atomic_load(&pool->stop))
                pthread_cond_wait(&pool->cv_work, &pool->mtx);
            pthread_mutex_unlock(&pool->mtx);
        }
        if (atomic_load(&pool->stop)) break;
        my_seq = atomic_load_explicit(&pool->seq, memory_order_acquire);

        run_range(pool->fn, pool->arg, pool->n, w->idx, pool->nworkers + 1, w->idx);

        /* completion: spin mode -> submitter polls `remaining`; condvar -> last worker signals cv_done */
        if (atomic_fetch_sub_explicit(&pool->remaining, 1, memory_order_acq_rel) == 1 && !pool->spin) {
            pthread_mutex_lock(&pool->mtx);
            pthread_cond_signal(&pool->cv_done);
            pthread_mutex_unlock(&pool->mtx);
        }
    }
    return NULL;
}

jam_pool* jam_pool_create(int nthreads) {
    if (nthreads < 1) nthreads = 1;
    jam_pool* pool = (jam_pool*) calloc(1, sizeof *pool);
    if (!pool) return NULL;
    pool->nworkers = nthreads - 1;   /* submitter is the +1 participant */
    const char* pm = getenv("JAM_POOL");
    pool->spin = (pm && strcmp(pm, "spin") == 0);
    const char* sb = getenv("JAM_SPIN");
    pool->spin_budget = sb ? atoi(sb) : 2048;
    if (pool->spin_budget < 1) pool->spin_budget = 1;
    pthread_mutex_init(&pool->mtx, NULL);
    pthread_cond_init(&pool->cv_work, NULL);
    pthread_cond_init(&pool->cv_done, NULL);
    if (pool->nworkers > 0) {
        pool->threads = (pthread_t*) calloc(pool->nworkers, sizeof(pthread_t));
        pool->wargs   = (jam_worker*) calloc(pool->nworkers, sizeof(jam_worker));
        for (int i = 0; i < pool->nworkers; ++i) {
            pool->wargs[i].pool = pool;
            pool->wargs[i].idx  = i + 1;   /* participants 1..nworkers; submitter is 0 */
            pthread_create(&pool->threads[i], NULL, worker_main, &pool->wargs[i]);
        }
    }
    return pool;
}

int jam_pool_is_spin(const jam_pool* pool)     { return pool ? pool->spin : 0; }
int jam_pool_spin_budget(const jam_pool* pool) { return pool ? pool->spin_budget : 0; }

void jam_pool_parallel_for(jam_pool* pool, int n, jam_task_fn fn, void* arg) {
    if (!pool || pool->nworkers == 0 || n <= 1) { fn(arg, 0, n, 0); return; }

    pool->fn = fn; pool->arg = arg; pool->n = n;
    atomic_store_explicit(&pool->remaining, pool->nworkers, memory_order_relaxed);

    if (pool->spin) {
        atomic_fetch_add_explicit(&pool->seq, 1, memory_order_seq_cst);   /* publish (also orders vs parked) */
        if (atomic_load_explicit(&pool->parked, memory_order_seq_cst) > 0) {
            pthread_mutex_lock(&pool->mtx);
            pthread_cond_broadcast(&pool->cv_work);                       /* wake any parked workers */
            pthread_mutex_unlock(&pool->mtx);
        }
        run_range(fn, arg, n, 0, pool->nworkers + 1, 0);                  /* submitter = participant 0 */
        while (atomic_load_explicit(&pool->remaining, memory_order_acquire) > 0) JAM_PAUSE();  /* spin-join */
    } else {
        pthread_mutex_lock(&pool->mtx);
        atomic_fetch_add(&pool->seq, 1);
        pthread_cond_broadcast(&pool->cv_work);
        pthread_mutex_unlock(&pool->mtx);

        run_range(fn, arg, n, 0, pool->nworkers + 1, 0);

        pthread_mutex_lock(&pool->mtx);
        while (atomic_load(&pool->remaining) > 0) pthread_cond_wait(&pool->cv_done, &pool->mtx);
        pthread_mutex_unlock(&pool->mtx);
    }
}

void jam_pool_destroy(jam_pool* pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mtx);
    atomic_store(&pool->stop, 1);
    pthread_cond_broadcast(&pool->cv_work);
    pthread_mutex_unlock(&pool->mtx);
    for (int i = 0; i < pool->nworkers; ++i) pthread_join(pool->threads[i], NULL);
    pthread_mutex_destroy(&pool->mtx);
    pthread_cond_destroy(&pool->cv_work);
    pthread_cond_destroy(&pool->cv_done);
    free(pool->threads);
    free(pool->wargs);
    free(pool);
}
