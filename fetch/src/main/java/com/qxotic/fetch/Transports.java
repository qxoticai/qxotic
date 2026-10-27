package com.qxotic.fetch;

import java.io.IOException;
import java.net.URI;
import java.nio.file.NoSuchFileException;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ScheduledThreadPoolExecutor;

/** The process-wide choices behind {@link Transport}, made lazily: at first use, not class load. */
final class Transports {
    static final Transport SYSTEM = choose();
    static final Transport OFFLINE =
            (uri, headers, timeout) -> {
                throw new NoSuchFileException("offline: no request to " + uri);
            };
    static final ScheduledThreadPoolExecutor DEADLINES = deadlines();

    private static Transport choose() {
        for (Transport provided : ServiceLoader.load(Transport.class)) {
            return provided;
        }
        if (CurlTransport.available()) {
            return new CurlTransport();
        }
        return (uri, headers, timeout) -> {
            throw new IOException(
                    "no HTTP transport: this build leaves out the JDK client and curl is not on"
                            + " the PATH; install curl to fetch "
                            + uri);
        };
    }

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

    /**
     * An http(s) URI with a host, or an IOException: anything else is a fetch failure, not a crash.
     */
    static void requireHttp(URI uri) throws IOException {
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        if (!scheme.equals("http") && !scheme.equals("https") || uri.getHost() == null)
            throw new IOException("not an http(s) URL: " + uri);
    }

    /** Request headers checked: a line break in a name or value is a smuggled request line. */
    static void requireClean(Map<String, String> headers) throws IOException {
        for (Map.Entry<String, String> header : headers.entrySet()) {
            String name = header.getKey();
            String value = header.getValue();
            if (name == null
                    || name.isBlank()
                    || name.contains(":")
                    || hasLineBreak(name)
                    || value == null
                    || hasLineBreak(value))
                throw new IOException("not a header: '" + name + ": " + value + "'");
        }
    }

    private static boolean hasLineBreak(String text) {
        return text.indexOf('\r') >= 0 || text.indexOf('\n') >= 0;
    }

    private Transports() {}
}
