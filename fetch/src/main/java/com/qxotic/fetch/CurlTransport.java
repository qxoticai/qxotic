package com.qxotic.fetch;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * {@code curl -sS -D - URL}: the headers arrive on stdout ahead of the body, so one stream carries
 * the whole response. Redirects are not followed ({@code -L} is not given). A transfer curl gives
 * up on is reported when the body reaches its end, so a truncated download cannot pass for a
 * complete one.
 *
 * <p>Hardening: no shell, every argument its own argv entry; {@code -q} so a user's curlrc cannot
 * redirect the transfer; the URL must be http or https, so it can never read as an option; request
 * headers travel in a config block on stdin, never in argv, so a token is not on the process list;
 * the timeout is a stall detector ({@code --speed-time}), the one curl has that spans headers.
 */
final class CurlTransport implements Transport {

    static boolean available() {
        try {
            Process probe =
                    new ProcessBuilder("curl", "--version").redirectErrorStream(true).start();
            probe.getInputStream().readAllBytes();
            return probe.waitFor(5, TimeUnit.SECONDS) && probe.exitValue() == 0;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    @Override
    public Response get(URI url, Map<String, String> headers, Duration timeout) throws IOException {
        String scheme = url.getScheme() == null ? "" : url.getScheme().toLowerCase(Locale.ROOT);
        if (!scheme.equals("http") && !scheme.equals("https"))
            throw new IOException("not an http(s) URL: " + url);
        long stall = Math.max(1, (timeout.toMillis() + 999) / 1000);
        List<String> command =
                new ArrayList<>(
                        List.of(
                                "curl",
                                "-q",
                                "-sS",
                                "--connect-timeout",
                                "30",
                                "--speed-time",
                                Long.toString(stall),
                                "--speed-limit",
                                "1",
                                "-K",
                                "-",
                                "-D",
                                "-",
                                "--",
                                url.toString()));
        Process process = new ProcessBuilder(command).start();
        StringBuilder config = new StringBuilder();
        headers.forEach(
                (name, value) ->
                        config.append("header = \"")
                                .append(
                                        (name + ": " + value)
                                                .replace("\\", "\\\\")
                                                .replace("\"", "\\\""))
                                .append("\"\n"));
        try (var stdin = process.getOutputStream()) {
            stdin.write(config.toString().getBytes(StandardCharsets.UTF_8));
        }
        InputStream out = new BufferedInputStream(process.getInputStream(), 1 << 16);

        int status;
        Map<String, List<String>> received = new HashMap<>();
        do { // a 1xx block, should one come, precedes the real one
            String statusLine = line(out);
            if (statusLine == null) throw failure(process, url);
            String[] parts = statusLine.split(" ", 3);
            try {
                status = Integer.parseInt(parts[1]);
            } catch (RuntimeException e) {
                throw new IOException(
                        "curl: unexpected status line '" + statusLine + "' from " + url);
            }
            received.clear();
            for (String header = line(out);
                    header != null && !header.isEmpty();
                    header = line(out)) {
                int colon = header.indexOf(':');
                if (colon > 0)
                    received.computeIfAbsent(
                                    header.substring(0, colon).strip().toLowerCase(Locale.ROOT),
                                    name -> new ArrayList<>())
                            .add(header.substring(colon + 1).strip());
            }
        } while (status / 100 == 1);

        int finalStatus = status;
        InputStream body =
                new FilterInputStream(out) {
                    private boolean ended;

                    @Override
                    public int read() throws IOException {
                        int b = super.read();
                        if (b < 0) ended();
                        return b;
                    }

                    @Override
                    public int read(byte[] b, int off, int len) throws IOException {
                        int n = super.read(b, off, len);
                        if (n < 0) ended();
                        return n;
                    }

                    private void ended() throws IOException {
                        if (ended) return;
                        ended = true;
                        try {
                            if (!process.waitFor(30, TimeUnit.SECONDS))
                                throw new IOException("curl did not exit after " + url);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IOException("interrupted while fetching " + url, e);
                        }
                        if (process.exitValue() != 0) throw failure(process, url);
                    }

                    @Override
                    public void close() throws IOException {
                        // destroy first: a reader blocked in the buffered stream holds its lock,
                        // and
                        // only curl's exit (pipe EOF) releases it; closing first would wait for
                        // that
                        process.destroy();
                        super.close();
                    }
                };
        return new Response() {
            @Override
            public int status() {
                return finalStatus;
            }

            @Override
            public String header(String name) {
                List<String> values = headers(name);
                return values.isEmpty() ? null : values.get(0);
            }

            @Override
            public List<String> headers(String name) {
                return received.getOrDefault(name.toLowerCase(Locale.ROOT), List.of());
            }

            @Override
            public InputStream body() {
                return body;
            }

            @Override
            public void close() throws IOException {
                body.close();
            }
        };
    }

    /** curl's own words from stderr, or its exit code when it had none. */
    private static IOException failure(Process process, URI url) throws IOException {
        String error =
                new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8).strip();
        try {
            process.waitFor(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return new IOException(
                error.isEmpty()
                        ? "curl exited with "
                                + (process.isAlive() ? "no response" : process.exitValue())
                                + " for "
                                + url
                        : error);
    }

    /** One header line, CRLF stripped; null at end of stream. */
    private static String line(InputStream in) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream(80);
        int b;
        while ((b = in.read()) >= 0 && b != '\n') bytes.write(b);
        if (b < 0 && bytes.size() == 0) return null;
        String line = bytes.toString(StandardCharsets.ISO_8859_1);
        return line.endsWith("\r") ? line.substring(0, line.length() - 1) : line;
    }
}
