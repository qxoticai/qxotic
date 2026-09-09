package com.qxotic.jinfer.codecs;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The one external-tool runner: streams any {@code stdin} on a daemon thread and drains stderr
 * concurrently so the child never blocks on a full pipe, bounds time and captured output, and
 * reports a non-zero exit with the captured stderr. Shared by the ffmpeg codecs and {@link Espeak}.
 */
final class Subprocess {

    private static final Duration TIMEOUT = Duration.ofMinutes(2);
    private static final int MAX_OUTPUT_BYTES = 256 << 20;
    private static final int MAX_ERROR_BYTES = 64 << 10;

    private Subprocess() {}

    static byte[] run(List<String> cmd, byte[] stdin) throws IOException {
        return run(cmd, stdin, TIMEOUT, MAX_OUTPUT_BYTES);
    }

    static byte[] run(List<String> cmd, byte[] stdin, Duration timeout, int maxOutputBytes)
            throws IOException {
        if (timeout == null || timeout.isNegative() || timeout.isZero()) {
            throw new IllegalArgumentException("timeout " + timeout);
        }
        if (maxOutputBytes < 1) {
            throw new IllegalArgumentException("maxOutputBytes " + maxOutputBytes);
        }
        String tool = Path.of(cmd.getFirst()).getFileName().toString();
        Process p;
        try {
            p = new ProcessBuilder(cmd).start();
        } catch (IOException e) {
            throw new IOException(
                    "failed to launch " + tool + " (is it on PATH?): " + e.getMessage(), e);
        }
        if (stdin != null) {
            daemon(
                    tool + "-stdin",
                    () -> {
                        try (OutputStream os = p.getOutputStream()) {
                            os.write(stdin);
                        } catch (IOException ignored) {
                            // Broken pipe if the tool rejects the input; its exit code reports it.
                        }
                    });
        } else {
            p.getOutputStream().close();
        }
        AtomicReference<byte[]> out = new AtomicReference<>();
        AtomicReference<IOException> outFailure = new AtomicReference<>();
        Thread outDrain =
                daemon(
                        tool + "-stdout",
                        () -> {
                            try (InputStream is = p.getInputStream()) {
                                out.set(readLimited(tool, is, maxOutputBytes));
                            } catch (IOException e) {
                                outFailure.set(e);
                            }
                        });
        ByteArrayOutputStream err = new ByteArrayOutputStream(Math.min(8192, MAX_ERROR_BYTES));
        Thread errDrain =
                daemon(
                        tool + "-stderr",
                        () -> {
                            try (InputStream es = p.getErrorStream()) {
                                drainCapped(es, err, MAX_ERROR_BYTES);
                            } catch (IOException ignored) {
                            }
                        });

        int code;
        try {
            if (!p.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS)) {
                p.destroyForcibly();
                p.waitFor();
                outDrain.join();
                errDrain.join();
                throw new IOException(tool + " timed out after " + timeout);
            }
            code = p.exitValue();
            outDrain.join();
            errDrain.join();
        } catch (InterruptedException e) {
            p.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("interrupted waiting for " + tool, e);
        }
        if (outFailure.get() != null) throw outFailure.get();
        if (code != 0) {
            throw new IOException(
                    tool + " exited " + code + ": " + err.toString(StandardCharsets.UTF_8).strip());
        }
        return out.get();
    }

    private static Thread daemon(String name, Runnable work) {
        Thread thread = new Thread(work, name);
        thread.setDaemon(true);
        thread.start();
        return thread;
    }

    private static byte[] readLimited(String tool, InputStream in, int limit) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(Math.min(8192, limit));
        byte[] buffer = new byte[8192];
        for (int read; (read = in.read(buffer)) >= 0; ) {
            int keep = Math.min(read, limit - out.size());
            out.write(buffer, 0, keep);
            if (keep != read) throw new IOException(tool + " output exceeds " + limit + " bytes");
        }
        return out.toByteArray();
    }

    private static void drainCapped(InputStream in, ByteArrayOutputStream out, int limit)
            throws IOException {
        byte[] buffer = new byte[8192];
        for (int read; (read = in.read(buffer)) >= 0; ) {
            int keep = Math.min(read, limit - out.size());
            if (keep > 0) out.write(buffer, 0, keep);
        }
    }
}
