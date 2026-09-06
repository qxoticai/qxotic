package com.qxotic.jinfer.testkit;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.OptionalLong;
import java.util.concurrent.TimeUnit;

/**
 * This JVM's resident set size in kilobytes, for footprint gates. Linux answers from {@code
 * /proc/self/status}, macOS from {@code ps}; on Windows and anywhere else the answer is empty, and
 * a gate {@code assume}s it away with {@link #unsupportedReason()} rather than failing.
 */
public final class ProcessRss {

    private static final String OS = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);

    private ProcessRss() {}

    /** Whether this platform can answer at all: Linux and macOS. */
    public static boolean supported() {
        return OS.contains("linux") || OS.contains("mac");
    }

    /** The one-line reason a gate skips here, naming the platform. */
    public static String unsupportedReason() {
        return "resident set is read on Linux (/proc) and macOS (ps) only, not on " + OS;
    }

    public static OptionalLong kilobytes() {
        if (!supported()) {
            return OptionalLong.empty();
        }
        try {
            return OS.contains("linux") ? fromProc() : fromPs();
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
        Process ps =
                new ProcessBuilder(
                                "ps",
                                "-o",
                                "rss=",
                                "-p",
                                Long.toString(ProcessHandle.current().pid()))
                        .redirectErrorStream(true)
                        .start();
        String out = new String(ps.getInputStream().readAllBytes()).strip();
        if (!ps.waitFor(10, TimeUnit.SECONDS) || ps.exitValue() != 0 || !out.matches("\\d+")) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(Long.parseLong(out));
    }
}
