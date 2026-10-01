package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LoadSpinnerTest {
    @Test
    void processShutdownEndsAnActiveLoadLine(@TempDir Path dir) throws Exception {
        var command = CliFixtures.javaCommand();
        command.addAll(
                List.of(
                        "-cp",
                        System.getProperty("java.class.path"),
                        ShutdownProbe.class.getName()));
        Path output = dir.resolve("shutdown.txt");
        Process process =
                new ProcessBuilder(command)
                        .redirectErrorStream(true)
                        .redirectOutput(output.toFile())
                        .start();
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "spinner shutdown hung");
            assertEquals(0, process.exitValue());
            String text = Files.readString(output);
            assertTrue(text.matches("(?s).*Loading model \\.{3,}\\R"), text);
        } finally {
            process.destroyForcibly();
        }
    }

    public static class ShutdownProbe {
        public static void main(String[] args) throws Exception {
            LoadSpinner.start("Loading model", System.err, true);
            if (args.length > 0) {
                System.out.println("ready");
                System.in.read();
            }
            System.exit(0); // bypasses close(), just as a signal during loading does
        }
    }

    @Test
    void interruptedCloseCannotLeaveADotAfterTheNewline() throws Exception {
        var bytes = new ByteArrayOutputStream();
        var pending = new CountDownLatch(1);
        var written = new CountDownLatch(1);
        var lineEnded = new CompletableFuture<Void>();
        var out =
                new PrintStream(bytes, true, StandardCharsets.UTF_8) {
                    @Override
                    public void print(char value) {
                        pending.countDown();
                        // Hold a dot while close runs. A correct close waits for this write.
                        lineEnded.completeOnTimeout(null, 1, TimeUnit.SECONDS).join();
                        super.print(value);
                        written.countDown();
                    }

                    @Override
                    public void println() {
                        super.println();
                        lineEnded.complete(null);
                    }
                };
        try (var spinner = LoadSpinner.start("Loading model", out, true)) {
            assertTrue(pending.await(5, TimeUnit.SECONDS));
            Thread.currentThread().interrupt();
            spinner.close();
            assertTrue(Thread.currentThread().isInterrupted());
        } finally {
            Thread.interrupted();
            lineEnded.complete(null);
        }
        assertTrue(written.await(5, TimeUnit.SECONDS));
        assertEquals(
                "Loading model ....\n",
                bytes.toString(StandardCharsets.UTF_8).replace("\r\n", "\n"));
    }

    @Test
    void redirectedOutputKeepsOnePlainLine() {
        var capture = new CliFixtures.Capture("");
        try (var spinner = LoadSpinner.start("Loading model", capture.io)) {
            assertEquals("Loading model ...\n", capture.err().replace("\r\n", "\n"));
        }
        assertEquals("Loading model ...\n", capture.err().replace("\r\n", "\n"));
        assertEquals("", capture.out());
    }

    @Test
    void dotsAppendAndClosingLeavesTheLineOnce() throws Exception {
        var bytes = new ByteArrayOutputStream();
        var frames = new CountDownLatch(2);
        var out =
                new PrintStream(bytes, true, StandardCharsets.UTF_8) {
                    @Override
                    public void print(char value) {
                        super.print(value);
                        if (value == '.') frames.countDown();
                    }
                };
        var spinner = LoadSpinner.start("Loading model", out, true);
        try (spinner) {
            assertTrue(bytes.toString(StandardCharsets.UTF_8).startsWith("Loading model ..."));
            assertTrue(frames.await(5, TimeUnit.SECONDS), "loading dots did not advance");
        }
        String output = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(output.replace("\r\n", "\n").matches("Loading model \\.{5,}\\n"), output);
        spinner.close();
        assertEquals(output, bytes.toString(StandardCharsets.UTF_8));
    }
}
