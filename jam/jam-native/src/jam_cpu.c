/* jam CPU core selection — see jam_cpu.h. The OS-independent policy is at the bottom; the per-OS
 * primitives (each best-effort, each with a safe default) are #ifdef'd above it. Only Linux is real
 * today; macOS/Windows fall back to "all online CPUs, unpinned" (the historical behaviour). */
#define _GNU_SOURCE
#include "jam_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================ per-OS primitives ============================
 * os_cpus_allowed(out)  -> count of usable logical CPU ids (online ∩ affinity ∩ cpuset)
 * os_cpu_capacity(id)   -> relative speed (freq kHz / DT capacity); 0 = unknown -> treated as top tier
 * os_cpu_core_key(id)   -> physical-core key; SMT siblings share it; unknown -> id (its own core)
 * os_cpu_quota_cores()  -> cgroup CPU quota in whole cores, 0 = none/unlimited
 */

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <unistd.h>

/* Read the first integer in a small sysfs/proc file; -1 if absent/unreadable. */
static long read_long(const char* path) {
    FILE* f = fopen(path, "re");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static int os_cpus_allowed(int* out) {
    cpu_set_t set;
    CPU_ZERO(&set);
    int n = 0;
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int id = 0; id < JAM_MAX_CPU && n < JAM_MAX_CPU; id++)
            if (CPU_ISSET(id, &set)) out[n++] = id;
    }
    if (n == 0) {                                   /* affinity unavailable -> all online */
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online < 1) online = 1;
        for (long id = 0; id < online && id < JAM_MAX_CPU; id++) out[n++] = (int) id;
    }
    return n;
}

static int os_cpu_capacity(int id) {
    char p[96];
    /* ARM exposes a normalized DT capacity (P=1024); x86 has no such file, so use max freq there. */
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpu_capacity", id);
    long c = read_long(p);
    if (c > 0) return (int) c;
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", id);
    c = read_long(p);
    return c > 0 ? (int) c : 0;                     /* 0 = unknown -> keep (top tier) */
}

static long os_cpu_core_key(int id) {
    char p[96];
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/core_id", id);
    long core = read_long(p);
    if (core < 0) return id;                         /* no topology -> treat as its own physical core */
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", id);
    long pkg = read_long(p);
    if (pkg < 0) pkg = 0;
    return pkg * 100000L + core;                     /* SMT siblings share (pkg, core_id) */
}

static int os_cpu_quota_cores(void) {
    /* cgroup v2 (unified): /sys/fs/cgroup/cpu.max = "<quota> <period>" or "max <period>". */
    FILE* f = fopen("/sys/fs/cgroup/cpu.max", "re");
    if (f) {
        long q = -1, period = 100000;
        char first[32] = {0};
        if (fscanf(f, "%31s %ld", first, &period) >= 1 && period > 0 && strcmp(first, "max") != 0)
            q = atol(first);
        fclose(f);
        if (q > 0) return (int) ((q + period - 1) / period);   /* ceil(quota/period) cores */
        return 0;
    }
    /* cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us (-1 quota = unlimited). */
    long q = read_long("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    long period = read_long("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
    if (q > 0 && period > 0) return (int) ((q + period - 1) / period);
    return 0;
}

int jam_cpu_pin(int cpu_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
}

int jam_cpu_can_pin(void) { return 1; }

#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread.h>     /* pthread_set_qos_class_self_np */

/* macOS gives no hard CPU affinity; the only real control is QoS (steer to the P "performance" cluster).
 * So we don't enumerate ids — we expose the P-core COUNT as the working set and steer each worker via QoS.
 * (The log therefore shows the P working set, not total logical; ids are nominal, QoS ignores them.) */
static int sysctl_int(const char* name, int dflt) {
    int v = 0; size_t sz = sizeof v;
    return (sysctlbyname(name, &v, &sz, NULL, 0) == 0 && v > 0) ? v : dflt;
}
static int os_cpus_allowed(int* out) {
    int phys = sysctl_int("hw.physicalcpu", 1);
    int p    = sysctl_int("hw.perflevel0.physicalcpu", phys);   /* Apple Silicon P-cores; Intel -> all phys */
    if (p > JAM_MAX_CPU) p = JAM_MAX_CPU;
    for (int i = 0; i < p; i++) out[i] = i;
    return p;
}
static int  os_cpu_capacity(int id)  { (void) id; return 0; }   /* already the P set */
static long os_cpu_core_key(int id)  { return id; }             /* already one-per-core (Apple = no SMT) */
static int  os_cpu_quota_cores(void) { return 0; }

int jam_cpu_pin(int cpu_id) {                                   /* soft steer to P-cores; cpu_id unused */
    (void) cpu_id;
    return pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) == 0;
}
int jam_cpu_can_pin(void) { return 1; }                         /* QoS steering counts as "pinned" */

#elif defined(_WIN32)
#include <windows.h>
/* EfficiencyClass (the P/E-core split, added in Win10) is the 2nd byte of PROCESSOR_RELATIONSHIP,
 * right after Flags. Older MinGW-w64 headers (Ubuntu CI) omit the named field, but the struct ABI is
 * fixed, so read it positionally - compiles on every toolchain and the real Windows SDK alike. */
#define JAM_EFFICIENCY_CLASS(proc) (((const BYTE*)&(proc))[1])

/* One in-affinity logical CPU per physical P-core (max EfficiencyClass), via GetLogicalProcessorInformationEx.
 * Single processor group (<=64 logical) — the common case; >64-CPU machines fall back to group 0 unpinned. */
static int os_cpus_allowed(int* out) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char* buf = (len > 0) ? (char*) malloc(len) : NULL;
    if (!buf || !GetLogicalProcessorInformationEx(RelationProcessorCore,
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) buf, &len)) { free(buf); return 0; }

    DWORD_PTR procmask = 0, sysmask = 0;
    GetProcessAffinityMask(GetCurrentProcess(), &procmask, &sysmask);

    BYTE best = 0;                                              /* pass 1: highest efficiency class = P */
    for (char* p = buf; p < buf + len; p += ((SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p)->Size) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
        if (e->Relationship == RelationProcessorCore && JAM_EFFICIENCY_CLASS(e->Processor) > best)
            best = JAM_EFFICIENCY_CLASS(e->Processor);
    }
    int n = 0;                                                  /* pass 2: one allowed sibling per P-core */
    for (char* p = buf; p < buf + len && n < JAM_MAX_CPU; p += ((SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p)->Size) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
        if (e->Relationship != RelationProcessorCore || JAM_EFFICIENCY_CLASS(e->Processor) != best) continue;
        KAFFINITY m = e->Processor.GroupMask[0].Mask;           /* group 0 only */
        for (int b = 0; b < 64; b++)
            if (((m >> b) & 1) && ((procmask >> b) & 1)) { out[n++] = b; break; }
    }
    free(buf);
    return n;
}
static int  os_cpu_capacity(int id)  { (void) id; return 0; }
static long os_cpu_core_key(int id)  { return id; }
static int  os_cpu_quota_cores(void) { return 0; }             /* job-object CPU rate caps: later */

int jam_cpu_pin(int cpu_id) {
    if (cpu_id >= 64) return 0;                                /* multi-group: best-effort skip */
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu_id) != 0;
}
int jam_cpu_can_pin(void) { return 1; }

#else  /* ---- unknown platform: safe defaults (all online CPUs, no P/E, no SMT split, no pin) ---- */
#include <unistd.h>
static int os_cpus_allowed(int* out) {
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1) online = 1;
    int n = 0;
    for (long id = 0; id < online && id < JAM_MAX_CPU; id++) out[n++] = (int) id;
    return n;
}
static int  os_cpu_capacity(int id)  { (void) id; return 0; }
static long os_cpu_core_key(int id)  { return id; }
static int  os_cpu_quota_cores(void) { return 0; }
int jam_cpu_pin(int cpu_id)  { (void) cpu_id; return 0; }
int jam_cpu_can_pin(void)    { return 0; }
#endif

/* ============================ OS-independent policy ============================ */

jam_cpu_plan jam_cpu_plan_make(void) {
    jam_cpu_plan p;
    memset(&p, 0, sizeof p);

    int allowed[JAM_MAX_CPU];
    int na = os_cpus_allowed(allowed);
    if (na < 1) { allowed[0] = 0; na = 1; }
    p.n_logical = na;

    /* Top capacity tier = "P-cores". maxcap==0 (unknown) -> thresh 0 -> everything qualifies. */
    int maxcap = 0;
    for (int i = 0; i < na; i++) {
        int c = os_cpu_capacity(allowed[i]);
        if (c > maxcap) maxcap = c;
    }
    int thresh = maxcap - maxcap / 8;                /* 7/8 of the fastest: separates P from E cores */

    long seen[JAM_MAX_CPU];
    int ns = 0;
    for (int i = 0; i < na; i++) {
        int id = allowed[i];
        if (os_cpu_capacity(id) < thresh) continue;  /* drop slow (E) cores */
        p.n_toptier++;
        long key = os_cpu_core_key(id);
        int dup = 0;
        for (int j = 0; j < ns; j++) if (seen[j] == key) { dup = 1; break; }
        if (dup) continue;                           /* drop SMT sibling of an already-picked core */
        seen[ns++] = key;
        p.cpu[p.n++] = id;
    }

    int q = os_cpu_quota_cores();                    /* cgroup: never exceed the bandwidth quota */
    if (q > 0 && p.n > q) { p.n = q; p.quota = q; }
    if (p.n < 1) { p.cpu[0] = 0; p.n = 1; }          /* paranoid floor */
    return p;
}
