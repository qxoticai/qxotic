package com.qxotic.jinfer;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;

/**
 * Runtime flags read by the parallel work runners and the model boundary. Prompt and cache defaults
 * live with their owning modules.
 */
public final class RuntimeFlags {

    /** Default scratch width for {@code newState}: prefill batches up to this many tokens. */
    public static final int BATCH_CAPACITY = positiveInt("jinfer.batchCapacity", 512);

    /**
     * Compute threads, prefill and decode alike: one per PHYSICAL core. Decode is memory-bandwidth
     * bound and a second SMT sibling only contends for the core's load/store ports; all-core
     * prefill measured +10-13% on the bench but P-core-only is simpler and cooler (user's call
     * 2026-08-27).
     */
    public static final int THREADS = positiveInt("jinfer.threads", physicalCoreCount());

    // Keys per flashDecode partition: below this there is nothing to gain from splitting the
    // attended range, so it falls through to rollingDecode.
    public static final int DECODE_BLOCK_SIZE = Integer.getInteger("jinfer.decodeBlockSize", 512);

    private static int positiveInt(String property, int defaultValue) {
        int value = Integer.getInteger(property, defaultValue);
        if (value < 1) {
            throw new IllegalArgumentException(property + " must be positive: " + value);
        }
        return value;
    }

    /**
     * Fast-core count for sizing the worker pools, measured where the platform can be asked: Apple
     * silicon P-cores via sysctl, Linux from sysfs topology (hybrid-, big.LITTLE- and
     * SMT-state-aware). Elsewhere (Windows) a heuristic: 2-way SMT on x86, none on ARM. Capped by
     * availableProcessors so a cgroup cpuset never gets oversubscribed (sysfs shows HOST cpus).
     * Override with -Djinfer.threads; read at run time so a native binary detects its host.
     */
    private static int physicalCoreCount() {
        int logical = Runtime.getRuntime().availableProcessors();
        int fast = applePerformanceCores();
        if (fast == 0) fast = linuxFastCores(Path.of("/sys"), Path.of("/proc/self/status"));
        if (fast > 0) return Math.min(fast, logical); // logical is cgroup-QUOTA-aware (cpu.max)
        String arch = System.getProperty("os.arch", "");
        boolean noSmt = arch.contains("aarch64") || arch.contains("arm");
        return noSmt ? logical : Math.max(1, logical / 2);
    }

    /**
     * Apple silicon P-core count via {@code sysctlbyname("hw.perflevel0.physicalcpu")}, or 0 when
     * not applicable. Decode barriers are latency-bound: one E-core straggler stalls every barrier,
     * and a gemma MoE token crosses ~1200 of them - measured 19.5 -> 27.1 t/s on an M3 Pro just by
     * excluding the E-cores. The jam backends run on this pool, so this count is theirs too.
     */
    private static int applePerformanceCores() {
        if (!System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("mac")
                || !System.getProperty("os.arch", "").contains("aarch64")) return 0;
        try {
            Linker linker = Linker.nativeLinker();
            var sysctl =
                    linker.downcallHandle(
                            linker.defaultLookup().find("sysctlbyname").orElseThrow(),
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_LONG));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment name = arena.allocateFrom("hw.perflevel0.physicalcpu");
                MemorySegment out = arena.allocate(ValueLayout.JAVA_INT);
                MemorySegment len = arena.allocate(ValueLayout.JAVA_LONG);
                len.set(ValueLayout.JAVA_LONG, 0, 4L);
                int rc = (int) sysctl.invokeExact(name, out, len, MemorySegment.NULL, 0L);
                return rc == 0 ? out.get(ValueLayout.JAVA_INT, 0) : 0;
            }
        } catch (Throwable t) {
            return 0; // no such sysctl (pre-perflevel macOS): the generic fallback decides
        }
    }

    /**
     * Linux fast-core count from sysfs, MEASURED rather than guessed: enumerate online CPUs,
     * restrict to the fast tier (Intel hybrid publishes the P-core list at {@code
     * devices/cpu_core/cpus}; ARM big.LITTLE ranks clusters by {@code cpu_capacity} - keep the top
     * tier; homogeneous parts have neither and keep everything), then count UNIQUE (package, core)
     * pairs - correct whether SMT is on, off (BIOS or nosmt), or absent, unlike a logical/2
     * heuristic. CONTAINER-aware: the walk starts from the cgroup/taskset affinity mask ({@code
     * Cpus_allowed_list} in /proc/self/status), so a cpuset-restricted pool never counts host cores
     * it cannot run on - and a cpuset pinned entirely OFF the fast tier falls back to the allowed
     * cores themselves (running on E-cores beats sizing a pool for absent P-cores). Package-visible
     * and rooted at caller paths so synthetic /sys + /proc trees can unit-test every topology and
     * confinement from any dev machine. 0 = not Linux / unreadable -> caller falls through to the
     * heuristic chain.
     */
    static int linuxFastCores(Path sys, Path procSelfStatus) {
        try {
            var cpus = parseCpuList(Files.readString(sys.resolve("devices/system/cpu/online")));
            List<Integer> allowed = allowedCpus(procSelfStatus);
            if (allowed != null) cpus.retainAll(allowed);
            List<Integer> fast = fastTier(sys, cpus);
            if (fast.isEmpty()) fast = cpus; // cpuset held no fast cores: run on what is allowed
            var cores = new HashSet<Long>();
            for (int c : fast) {
                Path t = sys.resolve("devices/system/cpu/cpu" + c + "/topology");
                long pkg =
                        Long.parseLong(Files.readString(t.resolve("physical_package_id")).trim());
                long core = Long.parseLong(Files.readString(t.resolve("core_id")).trim());
                cores.add(pkg << 32 | (core & 0xffffffffL));
            }
            return cores.size();
        } catch (Exception notLinuxOrOddSysfs) {
            return 0;
        }
    }

    /**
     * The fast-tier subset of {@code cpus}: Intel hybrid's published P-core list, else the top ARM
     * {@code cpu_capacity} tier, else all of them (homogeneous - no tier files).
     */
    private static List<Integer> fastTier(Path sys, List<Integer> cpus) throws Exception {
        Path hybrid = sys.resolve("devices/cpu_core/cpus"); // Intel hybrid: P-core cpu list
        if (Files.exists(hybrid)) {
            var p = new HashSet<>(parseCpuList(Files.readString(hybrid)));
            return cpus.stream().filter(p::contains).toList();
        }
        var capacity = new HashMap<Integer, Integer>();
        for (int c : cpus) {
            Path f = sys.resolve("devices/system/cpu/cpu" + c + "/cpu_capacity");
            if (!Files.exists(f)) return cpus; // x86/homogeneous: no capacity files
            capacity.put(c, Integer.parseInt(Files.readString(f).trim()));
        }
        if (capacity.isEmpty()) return cpus;
        int max = Collections.max(capacity.values());
        return cpus.stream().filter(c -> capacity.get(c) == max).toList();
    }

    /**
     * The process affinity set from {@code Cpus_allowed_list} (cgroup cpuset, docker --cpuset-cpus,
     * taskset), or null when unrestricted/unreadable (non-Linux, no such line).
     */
    private static List<Integer> allowedCpus(Path procSelfStatus) {
        try {
            for (String line : Files.readAllLines(procSelfStatus)) {
                if (line.startsWith("Cpus_allowed_list:")) {
                    return parseCpuList(line.substring("Cpus_allowed_list:".length()));
                }
            }
        } catch (Exception unreadable) {
            // fall through: treat as unrestricted
        }
        return null;
    }

    /** Kernel cpu-list syntax: {@code "0-3,8-11"} or {@code "0"}. */
    static List<Integer> parseCpuList(String list) {
        var out = new ArrayList<Integer>();
        for (String part : list.trim().split(",")) {
            if (part.isEmpty()) continue;
            int dash = part.indexOf('-');
            if (dash < 0) {
                out.add(Integer.parseInt(part));
            } else {
                int from = Integer.parseInt(part.substring(0, dash));
                int to = Integer.parseInt(part.substring(dash + 1));
                for (int i = from; i <= to; i++) out.add(i);
            }
        }
        return out;
    }

    private RuntimeFlags() {}
}
