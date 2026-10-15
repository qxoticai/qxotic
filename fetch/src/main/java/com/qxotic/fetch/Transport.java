package com.qxotic.fetch;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * One HTTP GET. Everything a hub download needs is here: the status, the headers (Range in,
 * Content-Range and Location out) and the body as a stream. Redirects are not followed by {@link
 * #get}; {@link #follow} does, dropping credentials when the host changes.
 *
 * <p>{@link #system()} is the transport a program should use: the JDK client when it is on the
 * class path, otherwise {@code curl} on the PATH. A native image excludes the JDK client with
 * {@code -H:ServiceLoaderFeatureExcludeServiceProviders=com.qxotic.fetch.JdkTransport} and carries
 * no TLS stack of its own; the system's does the work.
 */
public interface Transport {

    /** A response: the body is read by the caller and closed with the response. */
    interface Response extends Closeable {
        int status();

        /** The first value of a header, by case-insensitive name; null when absent. */
        String header(String name);

        InputStream body();
    }

    /**
     * One GET; a 3xx comes back as a 3xx. {@code timeout} bounds the wait for the status and
     * headers; the body is the caller's to bound, see {@link #deadline}.
     */
    Response get(URI url, Map<String, String> headers, Duration timeout) throws IOException;

    /**
     * A GET that follows up to five redirects. The Authorization header does not follow a redirect
     * to another host: the CDN URL is signed, and some CDNs refuse a credential alongside it.
     */
    default Response follow(URI url, Map<String, String> headers, Duration timeout)
            throws IOException {
        URI current = url;
        Map<String, String> currentHeaders = headers;
        for (int redirect = 0; ; redirect++) {
            Response response = get(current, currentHeaders, timeout);
            int status = response.status();
            if (status != 301 && status != 302 && status != 303 && status != 307 && status != 308) {
                return response;
            }
            String location = response.header("location");
            response.close();
            if (location == null || redirect == 5) {
                throw new IOException("redirect loop or missing Location for " + url);
            }
            URI next = current.resolve(location);
            if (!Objects.equals(current.getHost(), next.getHost())) {
                Map<String, String> stripped = new LinkedHashMap<>();
                currentHeaders.forEach(
                        (k, v) -> {
                            if (!k.equalsIgnoreCase("Authorization")) stripped.put(k, v);
                        });
                currentHeaders = stripped;
            }
            current = next;
        }
    }

    /**
     * The response, closed by force after {@code timeout} if the caller is still reading it: a
     * stalled body then fails its read instead of hanging the thread. Closing it earlier cancels.
     */
    static Response deadline(Response response, Duration timeout) {
        ScheduledFuture<?> close =
                System.DEADLINES.schedule(
                        () -> {
                            // on a thread of its own: a close that waits for the transfer to
                            // unwind must not hold up the other deadlines
                            Thread closer =
                                    new Thread(
                                            () -> {
                                                try {
                                                    response.close();
                                                } catch (IOException ignored) {
                                                    // the read already failed or completed
                                                }
                                            },
                                            "fetch-deadline-close");
                            closer.setDaemon(true);
                            closer.start();
                        },
                        timeout.toNanos(),
                        TimeUnit.NANOSECONDS);
        return new Response() {
            @Override
            public int status() {
                return response.status();
            }

            @Override
            public String header(String name) {
                return response.header(name);
            }

            @Override
            public InputStream body() {
                return response.body();
            }

            @Override
            public void close() throws IOException {
                close.cancel(false);
                response.close();
            }
        };
    }

    /** The JDK client when its provider is on the class path, otherwise curl on the PATH. */
    static Transport system() {
        return System.INSTANCE;
    }

    /** curl as a process: the system's HTTP, TLS, certificates and proxy settings. */
    static Transport curl() {
        return new CurlTransport();
    }

    /** Lazily: the choice is made at first use, not at class load. */
    final class System {
        static final Transport INSTANCE = choose();
        static final ScheduledThreadPoolExecutor DEADLINES = deadlines();

        private static ScheduledThreadPoolExecutor deadlines() {
            ScheduledThreadPoolExecutor executor =
                    new ScheduledThreadPoolExecutor(
                            1,
                            task -> {
                                Thread thread = new Thread(task, "fetch-deadline");
                                thread.setDaemon(true);
                                return thread;
                            });
            executor.setRemoveOnCancelPolicy(true);
            return executor;
        }

        private static Transport choose() {
            for (Transport provided : ServiceLoader.load(Transport.class)) {
                return provided;
            }
            if (CurlTransport.available()) {
                return new CurlTransport();
            }
            return (url, headers, timeout) -> {
                throw new IOException(
                        "no HTTP transport: this build leaves out the JDK client and curl is not on"
                                + " the PATH; install curl to fetch "
                                + url);
            };
        }

        private System() {}
    }
}
