package com.qxotic.jinfer.testkit;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.OptionalLong;
import java.util.concurrent.TimeUnit;

/**
 * This JVM's resident set size in kilobytes, for footprint gates. Linux answers from {@code
 * /proc/self/status}, macOS from {@code ps}, Windows from {@code tasklist}; anywhere else the
 * answer is empty, and a gate {@code assume}s it away with {@link #unsupportedReason()} rather than
 * failing.
 */
public final class ProcessRss {

    private static final String OS = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);

    private ProcessRss() {}

    /** Whether this platform can answer at all: Linux, macOS and Windows. */
    public static boolean supported() {
        return OS.contains("linux") || OS.contains("mac") || OS.contains("windows");
    }

    /** The one-line reason a gate skips here, naming the platform. */
    public static String unsupportedReason() {
        return "resident set is read on Linux (/proc), macOS (ps) and Windows (tasklist) only,"
                + " not on "
                + OS;
    }

    public static OptionalLong kilobytes() {
        if (!supported()) {
            return OptionalLong.empty();
        }
        try {
            if (OS.contains("linux")) return fromProc();
            return OS.contains("windows") ? fromTasklist() : fromPs();
        } catch (IOException | RuntimeException e) {
            return OptionalLong.empty();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return OptionalLong.empty();
        }
    }

    private static OptionalLong fromProc() throws IOException {
        for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
            if (line.startsWith("VmRSS:")) {
                return OptionalLong.of(Long.parseLong(line.replaceAll("[^0-9]", "")));
            }
        }
        return OptionalLong.empty();
    }

    /** {@code ps -o rss=} prints kilobytes on macOS (and the BSDs). */
    private static OptionalLong fromPs() throws IOException, InterruptedException {
        String out = run("ps", "-o", "rss=", "-p", Long.toString(ProcessHandle.current().pid()));
        return out != null && out.matches("\\d+")
                ? OptionalLong.of(Long.parseLong(out))
                : OptionalLong.empty();
    }

    /**
     * {@code tasklist} prints one CSV row, {@code "java.exe","1234","Console","1","123,456 K"}: the
     * working set in kilobytes, grouped with the locale's separator ({@code 123.456 K} on a German
     * Windows), so only the digits of the last field are read.
     */
    private static OptionalLong fromTasklist() throws IOException, InterruptedException {
        String out =
                run(
                        "tasklist",
                        "/FI",
                        "PID eq " + ProcessHandle.current().pid(),
                        "/FO",
                        "CSV",
                        "/NH");
        if (out == null || !out.startsWith("\"")) {
            return OptionalLong.empty(); // "INFO: No tasks are running..." on a miss
        }
        String last = out.substring(out.lastIndexOf(",\"") + 2).replaceAll("[^0-9]", "");
        return last.isEmpty() ? OptionalLong.empty() : OptionalLong.of(Long.parseLong(last));
    }

    /** The command's stripped output, or null when it fails or does not finish in time. */
    private static String run(String... command) throws IOException, InterruptedException {
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        String out = new String(process.getInputStream().readAllBytes()).strip();
        boolean done = process.waitFor(10, TimeUnit.SECONDS);
        if (!done) process.destroyForcibly();
        return done && process.exitValue() == 0 ? out : null;
    }
}
