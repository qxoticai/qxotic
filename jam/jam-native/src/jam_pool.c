/* A small persistent worker pool. Two barrier implementations, selectable per pool via JAM_POOL:
 *  - CONDVAR (default): workers park on a condvar between jobs — yields the CPU, ~µs wakeup.
 *  - SPIN (JAM_POOL=spin): workers busy-wait on the atomic job counter (~100 ns wakeup), falling back
 *    to parking after JAM_SPIN pauses so they don't burn cores when idle (e.g. during jinfer's Java
 *    decode). The submitter publishes via an atomic store and spins on `remaining` for the join. Wins
 *    when fan-outs come back-to-back (two-phase matmul, MoE small matmuls) — workers stay hot so the
 *    gap between fan-outs collapses.
 * The submitting thread participates as worker 0, so `nthreads` participants share each parallel_for. */
#include "jam_internal.h"
#include "jam_cpu.h"

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

typedef struct { jam_pool* pool; int idx; int cpu; } jam_worker;  /* cpu = pin target, -1 = float */

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
    int             participants; /* fan width for the current job (<= nworkers+1); see _capped */
    _Atomic uint32_t seq;         /* (generation << 16) | participants; change signals new work */
    _Atomic int     remaining;    /* workers still running the current job */
    _Atomic int     parked;       /* workers currently parked (spin mode: gate the wakeup broadcast) */
    _Atomic int     stop;
    int             nprimary;     /* participants 1..nprimary-1 are pinned one-per-physical-core */
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
    if (w->cpu >= 0) jam_cpu_pin(w->cpu);   /* best-effort: bind this worker to its selected core */
    uint32_t my_seq = 0;
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
        /* One atomic read = consistent (generation, fan width). Participants may read fn/arg/n:
         * their generation's barrier cannot complete without them, so the fields are pinned until
         * they finish. Non-participants read nothing and just adopt the generation. */
        uint32_t job = atomic_load_explicit(&pool->seq, memory_order_acquire);
        if (w->idx < (int)(job & 0xFFFF)) {
            int jpart = (int)(job & 0xFFFF);
            run_range(pool->fn, pool->arg, pool->n, w->idx, jpart, w->idx);
            /* completion: spin mode -> submitter polls `remaining`; condvar -> last participant signals */
            if (atomic_fetch_sub_explicit(&pool->remaining, 1, memory_order_acq_rel) == 1 && !pool->spin) {
                pthread_mutex_lock(&pool->mtx);
                pthread_cond_signal(&pool->cv_done);
                pthread_mutex_unlock(&pool->mtx);
            }
        }
        my_seq = job;
    }
    return NULL;
}

jam_pool* jam_pool_create(int nthreads, const int* cpu, int nprimary) {
    if (nthreads < 1) nthreads = 1;
    jam_pool* pool = (jam_pool*) calloc(1, sizeof *pool);
    if (!pool) return NULL;
    pool->nworkers = nthreads - 1;   /* submitter is the +1 participant */
    pool->nprimary = (nprimary >= 1 && nprimary <= nthreads) ? nprimary : nthreads;
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
            pool->wargs[i].cpu  = cpu ? cpu[i + 1] : -1;   /* cpu[0] is the (unpinned) submitter */
            pthread_create(&pool->threads[i], NULL, worker_main, &pool->wargs[i]);
        }
    }
    return pool;
}

int jam_pool_is_spin(const jam_pool* pool)     { return pool ? pool->spin : 0; }
int jam_pool_spin_budget(const jam_pool* pool) { return pool ? pool->spin_budget : 0; }

void jam_pool_parallel_for(jam_pool* pool, int n, jam_task_fn fn, void* arg) {
    jam_pool_parallel_for_capped(pool, n, fn, arg, 0);
}

/* As parallel_for, but at most `cap` participants share the range (0 = all). Workers beyond the cap
 * still cycle the barrier (empty range) so join accounting is unchanged. Bandwidth-bound phases cap
 * to one participant per physical core (the pinned-primary prefix). */
void jam_pool_parallel_for_capped(jam_pool* pool, int n, jam_task_fn fn, void* arg, int cap) {
    if (!pool || pool->nworkers == 0 || n <= 1) { fn(arg, 0, n, 0); return; }

    pool->fn = fn; pool->arg = arg; pool->n = n;
    int part = (cap > 0 && cap <= pool->nworkers) ? cap : pool->nworkers + 1;
    if (part > n) part = n;               /* never fan wider than the work units */
    if (part < 1) part = 1;
    /* Fan FULL-width (SMT-balanced: every core carries the same sibling count) or at most one
     * thread per physical core - never in between. A partial fan past nprimary lands 2 slices on
     * some cores and 1 on others (pin order is primaries-then-siblings), and the job runs at the
     * doubled cores' pace with the rest idle: measured 2x slower on MoE expert shapes. */
    if (part < pool->nworkers + 1 && part > pool->nprimary) part = pool->nprimary;
    /* participants ride INSIDE the seq word (gen<<16 | part): one atomic read hands a worker a
     * CONSISTENT (generation, fan-width) pair. A worker below the width dereferences fn/arg/n
     * freely - its generation's barrier cannot complete without it, so the fields are pinned; a
     * worker at/above the width never reads them at all. That closes the pre-bump tear where a
     * non-participant could catch the NEXT job's half-written fields under the OLD seq. */
    uint32_t next = ((atomic_load_explicit(&pool->seq, memory_order_relaxed) >> 16) + 1) << 16
                    | (uint32_t) part;
    /* only PARTICIPANT workers join the completion barrier: a small job on a wide pool must not
     * pay a 31-worker wake+join per call (MoE prefill issues thousands of tiny matmuls). Excluded
     * workers still observe the seq bump and cycle back to wait, off the critical path. */
    atomic_store_explicit(&pool->remaining, part - 1, memory_order_relaxed);

    if (pool->spin) {
        atomic_store_explicit(&pool->seq, next, memory_order_seq_cst);   /* publish (also orders vs parked) */
        if (atomic_load_explicit(&pool->parked, memory_order_seq_cst) > 0) {
            pthread_mutex_lock(&pool->mtx);
            pthread_cond_broadcast(&pool->cv_work);                       /* wake any parked workers */
            pthread_mutex_unlock(&pool->mtx);
        }
        run_range(fn, arg, n, 0, part, 0);                  /* submitter = participant 0 */
        while (atomic_load_explicit(&pool->remaining, memory_order_acquire) > 0) JAM_PAUSE();  /* spin-join */
    } else {
        pthread_mutex_lock(&pool->mtx);
        atomic_store(&pool->seq, next);
        pthread_cond_broadcast(&pool->cv_work);
        pthread_mutex_unlock(&pool->mtx);

        run_range(fn, arg, n, 0, part, 0);

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
