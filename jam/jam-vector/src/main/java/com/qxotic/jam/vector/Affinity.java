package com.qxotic.jam.vector;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pins Vector JAM's workers one per physical core (Linux only; a no-op elsewhere or on any
 * failure). Measured on a 16-core Zen 5: the same 16-thread sweep ran at 130 GF/s per thread with
 * the workers free to float and 230 pinned one-per-core - with equal CPU time, i.e. free workers
 * spent ~40% of the wall time descheduled or doubled up on SMT siblings after every park/unpark
 * between the pool's parallel phases. Only the workers are pinned; the caller, GC and JIT threads
 * stay free (pinning the whole JVM measured slower). {@code -Djam.vector.pin=false} disables it.
 */
final class Affinity {

    private Affinity() {}

    /** Logical CPUs to pin worker {@code i} to: element i, or null when pinning is off. */
    private static final int[] CPUS = plan();

    static boolean enabled() {
        return CPUS != null;
    }

    /** Pin the calling thread to the CPU planned for worker {@code index} (best effort). */
    static void pin(int index) {
        if (CPUS == null || index < 0 || index >= CPUS.length) return;
        try {
            MethodHandle setaffinity =
                    Linker.nativeLinker()
                            .downcallHandle(
                                    Linker.nativeLinker()
                                            .defaultLookup()
                                            .find("sched_setaffinity")
                                            .orElseThrow(),
                                    FunctionDescriptor.of(
                                            ValueLayout.JAVA_INT,
                                            ValueLayout.JAVA_INT,
                                            ValueLayout.JAVA_LONG,
                                            ValueLayout.ADDRESS));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment mask = arena.allocate(128, 8); // cpu_set_t: 1024 bits
                int cpu = CPUS[index];
                mask.set(
                        ValueLayout.JAVA_LONG,
                        (cpu / 64) * 8L,
                        mask.get(ValueLayout.JAVA_LONG, (cpu / 64) * 8L) | (1L << (cpu % 64)));
                int unused = (int) setaffinity.invokeExact(0, 128L, mask);
            }
        } catch (Throwable ignored) {
            // best effort: leave the thread unpinned
        }
    }

    /**
     * One logical CPU per worker: the process's allowed CPUs grouped by physical core, cores first
     * (one sibling each), then the remaining siblings. Null unless Linux, enabled, and the topology
     * is readable.
     */
    private static int[] plan() {
        if (!VectorSupport.jamProp("jam.vector.pin", "false").equals("true")) return null;
        if (!System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("linux"))
            return null;
        try {
            List<Integer> allowed = allowedCpus();
            if (allowed.isEmpty()) return null;
            Map<Integer, List<Integer>> byCore = new LinkedHashMap<>();
            for (int cpu : allowed) {
                int core =
                        Integer.parseInt(
                                read("/sys/devices/system/cpu/cpu" + cpu + "/topology/core_id"));
                int pkg =
                        Integer.parseInt(
                                read(
                                        "/sys/devices/system/cpu/cpu"
                                                + cpu
                                                + "/topology/physical_package_id"));
                byCore.computeIfAbsent(pkg << 16 | core, k -> new ArrayList<>()).add(cpu);
            }
            List<Integer> order = new ArrayList<>();
            for (int round = 0; order.size() < allowed.size(); round++)
                for (List<Integer> siblings : byCore.values())
                    if (round < siblings.size()) order.add(siblings.get(round));
            int n = Math.min(VectorSupport.PARALLELISM, order.size());
            if (VectorSupport.PARALLELISM > byCore.size())
                return null; // more workers than cores: float
            int[] cpus = new int[n];
            for (int i = 0; i < n; i++) cpus[i] = order.get(i);
            return cpus;
        } catch (Throwable t) {
            return null;
        }
    }

    /** CPUs this process may run on, from /proc/self/status Cpus_allowed_list. */
    private static List<Integer> allowedCpus() throws java.io.IOException {
        List<Integer> cpus = new ArrayList<>();
        for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
            if (!line.startsWith("Cpus_allowed_list:")) continue;
            for (String range : line.substring("Cpus_allowed_list:".length()).trim().split(",")) {
                String[] lohi = range.trim().split("-");
                int lo = Integer.parseInt(lohi[0]), hi = Integer.parseInt(lohi[lohi.length - 1]);
                for (int c = lo; c <= hi; c++) cpus.add(c);
            }
        }
        return cpus;
    }

    private static String read(String path) throws java.io.IOException {
        return Files.readString(Path.of(path)).trim();
    }
}
