package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

/** The bar is rendered output - pinned as strings, like any other behavior. */
class FetchBarTest {

    @ParameterizedTest
    @ValueSource(ints = {30, 50, 80})
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void progressFitsStderrWithStdinRedirected(int columns, @TempDir Path dir) throws Exception {
        // A real output PTY, redirected stdin, and a stale COLUMNS inherited from another terminal.
        String script =
                """
                import errno, fcntl, os, pty, struct, subprocess, sys, termios
                master, slave = pty.openpty()
                fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 12, int(sys.argv[1]), 0, 0))
                env = dict(os.environ, TERM='xterm', COLUMNS='80')
                env.pop('NO_COLOR', None)
                child = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=slave, env=env)
                os.close(slave)
                try:
                    while True:
                        try:
                            data = os.read(master, 65536)
                        except OSError as error:
                            if error.errno != errno.EIO:
                                raise
                            break
                        if not data:
                            break
                        sys.stdout.buffer.write(data)
                    sys.exit(child.wait())
                finally:
                    os.close(master)
                    if child.poll() is None:
                        child.kill()
                        child.wait()
                """;
        List<String> command =
                new ArrayList<>(List.of("python3", "-c", script, Integer.toString(columns)));
        command.addAll(probeCommand());
        Path output = dir.resolve("terminal.txt"), error = dir.resolve("stderr.txt");
        Process process;
        try {
            process =
                    new ProcessBuilder(command)
                            .redirectOutput(output.toFile())
                            .redirectError(error.toFile())
                            .start();
        } catch (IOException unavailable) {
            Assumptions.abort("python3 is required for the PTY check: " + unavailable.getMessage());
            return;
        }
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "PTY probe timed out");
            assertEquals(0, process.exitValue(), Files.readString(error));
            String capture = Files.readString(output);
            var rows = Pattern.compile("\r\u001b\\[2K([^\r\n]*)\r?\n").matcher(capture);
            int count = 0;
            boolean completed = false;
            while (rows.find()) {
                String row = rows.group(1);
                assertTrue(
                        row.length() <= columns - 1, "row exceeds " + columns + " columns: " + row);
                completed |= row.startsWith(" ✓ ") || row.startsWith(" * ");
                count++;
            }
            assertTrue(count >= 2, "expected live and completed progress rows: " + capture);
            assertTrue(completed, "the completed download must remain visible: " + capture);
            assertFalse(capture.contains("stty:"), "width probing must be silent: " + capture);
        } finally {
            process.descendants().forEach(ProcessHandle::destroyForcibly);
            process.destroyForcibly();
        }
    }

    private static List<String> probeCommand() throws Exception {
        String classes =
                Path.of(Fetch.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                        .toString();
        return List.of(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--enable-native-access=ALL-UNNAMED",
                "-cp",
                classes + File.pathSeparator + System.getProperty("java.class.path"),
                ProgressProbe.class.getName());
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"17", "0", "-1", "invalid"})
    void missingTerminalUsesAValidEnvironmentWidthOr80(String columns, @TempDir Path dir)
            throws Exception {
        List<String> command = new ArrayList<>(probeCommand());
        command.add("columns");
        Path output = dir.resolve("width.txt"), error = dir.resolve("stderr.txt");
        var builder =
                new ProcessBuilder(command)
                        .redirectOutput(output.toFile())
                        .redirectError(error.toFile());
        if (columns == null) builder.environment().remove("COLUMNS");
        else builder.environment().put("COLUMNS", columns);
        Process process = builder.start();
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "width probe timed out");
            assertEquals(0, process.exitValue(), Files.readString(error));
            assertEquals("17".equals(columns) ? "17" : "80", Files.readString(output).strip());
            String diagnostics = Files.readString(error);
            assertFalse(diagnostics.contains("stty:"), diagnostics);
            assertFalse(diagnostics.contains("/bin/sh:"), diagnostics);
        } finally {
            process.descendants().forEach(ProcessHandle::destroyForcibly);
            process.destroyForcibly();
        }
    }

    /** A log gets two lines per download, start and done, never the bar's frames. */
    @Test
    void redirectedProgressIsAStartLineAndADoneLine(@TempDir Path dir) throws Exception {
        Path output = dir.resolve("stdout.txt"), error = dir.resolve("stderr.txt");
        var builder =
                new ProcessBuilder(probeCommand())
                        .redirectOutput(output.toFile())
                        .redirectError(error.toFile());
        builder.environment().remove("NO_COLOR");
        Process process = builder.start();
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "probe timed out");
            assertEquals(0, process.exitValue(), Files.readString(error));
            List<String> lines =
                    Files.readString(error).lines().filter(l -> l.contains("model-Q8_0")).toList();
            assertEquals(2, lines.size(), Files.readString(error));
            assertTrue(lines.getFirst().endsWith("model-Q8_0.gguf  1000 B"), lines.getFirst());
            assertTrue(
                    lines.getLast().contains("model-Q8_0.gguf  1000 B done in "), lines.getLast());
            assertFalse(Files.readString(error).contains("eta"), Files.readString(error));
        } finally {
            process.destroyForcibly();
        }
    }

    public static class ProgressProbe {
        public static void main(String[] args) throws Exception {
            if (args.length > 0) {
                System.out.println(TerminalSupport.columns(2));
                return;
            }
            Fetch.Progress progress = new Fetch.Progress("model-Q8_0.gguf", 1000);
            progress.start(0);
            progress.at(500);
            progress.finish();
        }
    }

    @Test
    void nativeAccessIsOptionalForPlainOutput(@TempDir Path dir) throws Exception {
        List<String> command = new ArrayList<>(probeCommand());
        command.remove("--enable-native-access=ALL-UNNAMED");
        command.add(1, "--illegal-native-access=deny");
        command.add("columns");
        Path output = dir.resolve("width.txt"), error = dir.resolve("stderr.txt");
        var builder =
                new ProcessBuilder(command)
                        .redirectOutput(output.toFile())
                        .redirectError(error.toFile());
        // This probe intentionally denies access, independent of the test runner's launch flags.
        for (String variable : List.of("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"))
            builder.environment().remove(variable);
        builder.environment().put("COLUMNS", "17");
        Process process = builder.start();
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS));
            assertEquals(0, process.exitValue(), Files.readString(error));
            assertEquals("17", Files.readString(output).strip());
            assertEquals("", Files.readString(error));
        } finally {
            process.destroyForcibly();
        }
    }

    @Test
    void subCellEdgeGlides() {
        // 4 cells wide: 1/8 of a cell shows the thinnest block, and each eighth advances it
        assertEquals("    ", Fetch.Progress.bar(0, 800, 4, true));
        assertEquals("▏   ", Fetch.Progress.bar(25, 800, 4, true));
        assertEquals("▌   ", Fetch.Progress.bar(100, 800, 4, true));
        assertEquals("▉   ", Fetch.Progress.bar(175, 800, 4, true));
        assertEquals("█   ", Fetch.Progress.bar(200, 800, 4, true));
        assertEquals("██▌ ", Fetch.Progress.bar(500, 800, 4, true));
        assertEquals("████", Fetch.Progress.bar(800, 800, 4, true));
        assertEquals("████", Fetch.Progress.bar(900, 800, 4, true)); // over-report never overflows
    }

    @Test
    void asciiConsolesKeepWholeCells() {
        assertEquals("----", Fetch.Progress.bar(25, 800, 4, false));
        assertEquals("#---", Fetch.Progress.bar(200, 800, 4, false));
        assertEquals(
                "##--", Fetch.Progress.bar(500, 800, 4, false)); // 5/8 truncates, never rounds up
        assertEquals("####", Fetch.Progress.bar(800, 800, 4, false));
    }
}
