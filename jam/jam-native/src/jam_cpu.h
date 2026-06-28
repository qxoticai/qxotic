/* jam CPU core selection — INTERNAL, best-effort. NOT public API; no stability guarantees.
 *
 * Picks which logical CPUs jam's pool should run on, by three filters on the OS-allowed set:
 *     allowed CPUs -> keep the top capacity tier (P-cores) -> one per physical core (drop SMT)
 *                  -> cap the count by the cgroup CPU quota
 * Every per-OS primitive has a "don't know -> safe default" so this NEVER fails: an unknown platform
 * just yields "all allowed logical CPUs, unpinned" (the historical behaviour). The chosen plan is meant
 * to be DUMPED TO THE LOG (jam.c) so a user can see what was selected. The user-facing knobs come later;
 * for now JAM_NUM_THREADS still overrides the count (see jam.c). */
#ifndef JAM_CPU_H
#define JAM_CPU_H

#define JAM_MAX_CPU 512   /* selection caps here; bigger machines just see the first 512 logical CPUs */

typedef struct {
    int n;                  /* selected count = default pool size */
    int cpu[JAM_MAX_CPU];   /* selected logical CPU ids, one per physical P-core */
    int pinned;             /* 1 if the plan was handed to the pool to pin (best-effort), else 0 */
    /* summary, for the log line only */
    int n_logical;          /* allowed logical CPUs the OS gave us */
    int n_toptier;          /* of those, how many are top-capacity-tier (the "P" cores) */
    int quota;              /* cgroup CPU quota in cores that capped n, or 0 if none */
} jam_cpu_plan;

/* Build the selection plan (detect + filter). Pure best-effort; always returns a usable plan (n>=1). */
jam_cpu_plan jam_cpu_plan_make(void);

/* Pin the CURRENT thread to one logical CPU. Best-effort: 1 on success, 0 if unsupported/failed.
 * Called by each pool worker at start-up; the submitter (caller/JVM thread) is never pinned. */
int jam_cpu_pin(int cpu_id);

/* 1 on platforms where jam_cpu_pin can actually bind a thread (so the log reports "pinned" honestly). */
int jam_cpu_can_pin(void);

#endif /* JAM_CPU_H */
