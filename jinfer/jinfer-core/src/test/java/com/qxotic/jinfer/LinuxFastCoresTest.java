package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link RuntimeFlags#linuxFastCores} against synthetic sysfs trees - one per real topology, so the
 * Linux fast-core logic is verified on any dev machine. Each tree writes only the files the
 * detector reads: {@code devices/system/cpu/online}, optional {@code devices/cpu_core/cpus} (Intel
 * hybrid), optional per-cpu {@code cpu_capacity} (ARM big.LITTLE), and per-cpu {@code
 * topology/physical_package_id} + {@code core_id}.
 */
class LinuxFastCoresTest {

    @TempDir Path sys;

    /** cpu c on package pkg, core id core, with optional capacity. */
    private void cpu(int c, int pkg, int core, Integer capacity) throws IOException {
        Path d = sys.resolve("devices/system/cpu/cpu" + c);
        Files.createDirectories(d.resolve("topology"));
        Files.writeString(d.resolve("topology/physical_package_id"), pkg + "\n");
        Files.writeString(d.resolve("topology/core_id"), core + "\n");
        if (capacity != null) Files.writeString(d.resolve("cpu_capacity"), capacity + "\n");
    }

    /** An absent /proc/self/status: unrestricted affinity. */
    private Path noProc() {
        return sys.resolve("no-proc-status");
    }

    /** A synthetic /proc/self/status carrying the given affinity list. */
    private Path proc(String cpusAllowedList) throws IOException {
        Path f = sys.resolve("proc-status");
        Files.writeString(
                f, "Name:\tjava\nCpus_allowed_list:\t" + cpusAllowedList + "\nSeccomp:\t0\n");
        return f;
    }

    private void online(String list) throws IOException {
        Files.createDirectories(sys.resolve("devices/system/cpu"));
        Files.writeString(sys.resolve("devices/system/cpu/online"), list + "\n");
    }

    /** 8 P-cores (cpus 0-15, SMT pairs) + 8 E-cores (cpus 16-23), cpu_core lists the P set. */
    private void hybridHost() throws IOException {
        online("0-23");
        for (int c = 0; c < 16; c++) cpu(c, 0, c / 2, null); // P: cpus 0-15, cores 0-7
        for (int c = 16; c < 24; c++) cpu(c, 0, 8 + (c - 16), null); // E: cores 8-15
        Files.createDirectories(sys.resolve("devices/cpu_core"));
        Files.writeString(sys.resolve("devices/cpu_core/cpus"), "0-15\n");
    }

    @Test
    void homogeneousWithSmt() throws IOException { // 4 cores x 2 threads, siblings interleaved
        online("0-7");
        for (int c = 0; c < 8; c++) cpu(c, 0, c % 4, null);
        assertEquals(4, RuntimeFlags.linuxFastCores(sys, noProc()));
    }

    @Test
    void homogeneousSmtDisabled() throws IOException { // nosmt: one online thread per core
        online("0-3");
        for (int c = 0; c < 4; c++) cpu(c, 0, c, null);
        assertEquals(
                4,
                RuntimeFlags.linuxFastCores(sys, noProc())); // NOT halved (the old heuristic's bug)
    }

    @Test
    void intelHybrid() throws IOException {
        hybridHost();
        assertEquals(8, RuntimeFlags.linuxFastCores(sys, noProc()));
    }

    @Test
    void armBigLittle() throws IOException { // 4 big (1024) + 4 little (512), no SMT
        online("0-7");
        for (int c = 0; c < 4; c++) cpu(c, 0, c, 1024);
        for (int c = 4; c < 8; c++) cpu(c, 0, c, 512);
        assertEquals(4, RuntimeFlags.linuxFastCores(sys, noProc()));
    }

    @Test
    void cpusetRestrictsToAllowedFastCores() throws IOException { // container on hybrid host
        hybridHost();
        // --cpuset-cpus=0-3,16-19: two P-cores (4 SMT threads) + four E-cores
        assertEquals(2, RuntimeFlags.linuxFastCores(sys, proc("0-3,16-19")));
    }

    @Test
    void cpusetPinnedEntirelyToECores() throws IOException { // fast tier empty -> allowed cores
        hybridHost();
        assertEquals(4, RuntimeFlags.linuxFastCores(sys, proc("16-19"))); // 4 E-cores, not 0
    }

    @Test
    void statusWithoutAffinityLineIsUnrestricted() throws IOException {
        online("0-3");
        for (int c = 0; c < 4; c++) cpu(c, 0, c, null);
        Path f = sys.resolve("proc-status");
        Files.writeString(f, "Name:\tjava\n");
        assertEquals(4, RuntimeFlags.linuxFastCores(sys, f));
    }

    @Test
    void unreadableTreeFallsThrough() {
        assertEquals(0, RuntimeFlags.linuxFastCores(sys.resolve("nonexistent"), noProc()));
    }

    @Test
    void cpuListSyntax() {
        assertEquals(List.of(0), RuntimeFlags.parseCpuList("0"));
        assertEquals(List.of(0, 1, 2, 3, 8, 9), RuntimeFlags.parseCpuList("0-3,8-9"));
    }
}
