package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.message.UserMessage;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The in-fork memory-growth forensic: cycles battery-like model load/chat/close in ONE JVM and
 * prints, per cycle, RssAnon/RssFile, the NMT category deltas, and the effect of malloc_trim(0).
 * Decision tree: NMT "Other" growing = an FFM allocation we failed to free; NMT flat but RssAnon
 * growing = native memory NMT cannot see (jam's own allocations) or glibc-retained freed pages -
 * and malloc_trim collapsing RssAnon proves the latter. Run with:
 *
 * <pre>
 * JAVA_TOOL_OPTIONS=-XX:NativeMemoryTracking=summary mvn -f langchain4j-jinfer/pom.xml test \
 *   -Dsurefire.excludedGroups= -Dgroups=bench -Dtest=NmtProbe
 * </pre>
 */
@Tag("bench")
class NmtProbe {

    static final MethodHandle MALLOC_TRIM =
            Linker.nativeLinker()
                    .downcallHandle(
                            Linker.nativeLinker().defaultLookup().find("malloc_trim").orElseThrow(),
                            FunctionDescriptor.of(
                                    java.lang.foreign.ValueLayout.JAVA_INT,
                                    java.lang.foreign.ValueLayout.JAVA_INT));

    record Cycle(String name, Path path) {}

    @Test
    void loadChatCloseRotation() throws Throwable {
        List<Cycle> rotation =
                List.of(
                        new Cycle("lfm2-8b-moe", ModelFixture.LFM25_8B_Q8.path()),
                        new Cycle("granite-3b", ModelFixture.GRANITE_41_3B_Q8.path()),
                        new Cycle("qwen35-2b", ModelFixture.QWEN35_2B_Q8.path()),
                        new Cycle("minicpm5-1b", ModelFixture.MINICPM5_1B_Q8.path()),
                        new Cycle("lfm2-350m", ModelFixture.LFM25_350M_Q8.path()));
        System.out.printf(
                "%-14s %9s %9s %9s %9s %9s %9s %9s%n",
                "cycle",
                "anonMB",
                "fileMB",
                "nmtTotMB",
                "otherMB",
                "heapMB",
                "threadMB",
                "trimdMB");
        snapshot("baseline", 0);
        for (int round = 0; round < 2; round++) {
            for (Cycle c : rotation) {
                if (!Files.exists(c.path())) {
                    System.out.println(c.name() + ": model absent, skipped");
                    continue;
                }
                try (JinferChatModel m =
                        JinferChatModel.builder()
                                .modelPath(c.path())
                                .contextLength(4096)
                                .maxOutputTokens(16)
                                .build()) {
                    m.chat(UserMessage.from("Say hi in one word."));
                }
                snapshot(c.name() + "#" + round, 0);
            }
        }
        // the levers, in order of coercion: trim glibc, then a full GC + cleaner pass
        long trimmed = trim();
        snapshot("post-trim", trimmed);
        System.gc();
        Thread.sleep(500);
        long trimmed2 = trim();
        snapshot("post-gc+trim", trimmed2);
    }

    static long trim() throws Throwable {
        long before = rss("RssAnon");
        MALLOC_TRIM.invoke(0);
        return before - rss("RssAnon");
    }

    static void snapshot(String label, long trimmedKb) throws Exception {
        long anon = rss("RssAnon") / 1024, file = rss("RssFile") / 1024;
        long[] nmt = nmt();
        System.out.printf(
                "%-14s %9d %9d %9d %9d %9d %9d %9d%n",
                label, anon, file, nmt[0], nmt[1], nmt[2], nmt[3], trimmedKb / 1024);
    }

    static long rss(String key) throws Exception {
        for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
            if (line.startsWith(key + ":")) return Long.parseLong(line.replaceAll("[^0-9]", ""));
        }
        return -1;
    }

    /** total-committed, Other-committed, JavaHeap-committed, Thread-committed - MB. */
    static long[] nmt() throws Exception {
        Process p =
                new ProcessBuilder(
                                "jcmd",
                                String.valueOf(ProcessHandle.current().pid()),
                                "VM.native_memory",
                                "summary")
                        .redirectErrorStream(true)
                        .start();
        long[] out = new long[4];
        String category = "";
        for (String line : new String(p.getInputStream().readAllBytes()).split("\n")) {
            String t = line.strip();
            if (t.startsWith("Total:")) {
                out[0] = committedKb(t) / 1024;
            } else if (t.startsWith("-")) {
                category = t;
                long kb = committedKb(t);
                if (t.contains("Other")) out[1] = kb / 1024;
                else if (t.contains("Java Heap")) out[2] = kb / 1024;
                else if (t.contains("Thread")) out[3] = kb / 1024;
            }
        }
        p.waitFor();
        return out;
    }

    static long committedKb(String line) {
        // formats: "committed=123456KB" (Total) / "(committed=123456KB)" (categories)
        int i = line.indexOf("committed=");
        if (i < 0) return 0;
        int end = line.indexOf("KB", i);
        if (end < 0) return 0;
        return Long.parseLong(line.substring(i + "committed=".length(), end).trim());
    }
}
