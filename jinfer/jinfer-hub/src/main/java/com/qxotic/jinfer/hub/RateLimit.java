package com.qxotic.jinfer.hub;

import java.net.http.HttpHeaders;
import java.time.Duration;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Server rate limits, honored: how long a 429 asks us to wait, and a per-host hold that every
 * request to that host waits out. The Hugging Face CDN answers a burst with {@code 429} and the
 * IETF header {@code RateLimit: "resolvers";r=0;t=21} ("nothing left, the window resets in 21 s"),
 * per IP across every file in flight - so retrying at once, from every chunk worker, only spends
 * requests the window has not given back yet. One hold per host makes the whole process pause
 * together and resume when the server said it may.
 */
final class RateLimit {

    /** The longest single wait honored; a server asking for more is treated as asking for this. */
    static final Duration MAX_WAIT = Duration.ofMinutes(15);

    /** The total a single request may spend waiting before its 429 is reported as a failure. */
    static final Duration BUDGET = Duration.ofMinutes(30);

    private static final Map<String, Long> HOLD_UNTIL = new ConcurrentHashMap<>(); // nanoTime

    // RateLimit structured field: `"policy";r=0;t=21`, several comma-separated; t = seconds to
    // reset
    private static final Pattern ITEM_T = Pattern.compile("(?:^|;)\\s*t\\s*=\\s*(\\d+)");
    private static final Pattern ITEM_R = Pattern.compile("(?:^|;)\\s*r\\s*=\\s*(\\d+)");
    // the older draft's dictionary form: `limit=3000, remaining=0, reset=21`
    private static final Pattern DICT_RESET = Pattern.compile("(?:^|,)\\s*reset\\s*=\\s*(\\d+)");

    private RateLimit() {}

    /**
     * The wait the response asks for, or null when it names none: {@code Retry-After} (seconds or
     * an HTTP date), else the {@code RateLimit} reset of an exhausted policy, else {@code
     * RateLimit-Reset}.
     */
    static Duration requested(HttpHeaders headers, long nowEpochMillis) {
        String retryAfter = headers.firstValue("retry-after").orElse(null);
        if (retryAfter != null) {
            Duration d = retryAfter(retryAfter.strip(), nowEpochMillis);
            if (d != null) return d;
        }
        long reset = -1;
        for (String field : headers.allValues("ratelimit")) {
            for (String item : field.split(",")) {
                Matcher t = ITEM_T.matcher(item);
                if (!t.find()) continue;
                Matcher r = ITEM_R.matcher(item);
                if (r.find() && Long.parseLong(r.group(1)) > 0) continue; // this policy has room
                reset = Math.max(reset, Long.parseLong(t.group(1)));
            }
            Matcher dict = DICT_RESET.matcher(field);
            if (reset < 0 && dict.find()) reset = Long.parseLong(dict.group(1));
        }
        if (reset < 0) {
            reset = headers.firstValue("ratelimit-reset").map(RateLimit::seconds).orElse(-1L);
        }
        return reset < 0 ? null : Duration.ofSeconds(reset);
    }

    private static Duration retryAfter(String value, long nowEpochMillis) {
        long s = seconds(value);
        if (s >= 0) return Duration.ofSeconds(s);
        try {
            long at =
                    ZonedDateTime.parse(value, DateTimeFormatter.RFC_1123_DATE_TIME)
                            .toInstant()
                            .toEpochMilli();
            return Duration.ofMillis(Math.max(0, at - nowEpochMillis));
        } catch (DateTimeParseException e) {
            return null;
        }
    }

    private static long seconds(String value) {
        try {
            return Long.parseLong(value.strip());
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /**
     * Holds {@code host} for {@code wait} from now. Returns true when this call moved the hold
     * later - the one caller that should say so, rather than every worker that hit the same wall.
     */
    static boolean hold(String host, Duration wait) {
        long until = System.nanoTime() + wait.toNanos();
        boolean[] extended = {false};
        HOLD_UNTIL.compute(
                host,
                (h, old) -> {
                    // a second of slack: workers hitting the same wall a few ms apart are one hold
                    if (old == null || until > old + 1_000_000_000L) {
                        extended[0] = true;
                        return old == null ? until : Math.max(old, until);
                    }
                    return Math.max(old, until);
                });
        return extended[0];
    }

    /** Blocks until {@code host} is not held. Returns the nanoseconds spent waiting. */
    static long awaitClear(String host) throws InterruptedException {
        long start = System.nanoTime();
        while (true) {
            Long until = HOLD_UNTIL.get(host);
            long now = System.nanoTime();
            if (until == null || now >= until) {
                if (until != null) HOLD_UNTIL.remove(host, until);
                return now - start;
            }
            Thread.sleep(Math.max(1, Math.min((until - now) / 1_000_000L, 1_000)));
        }
    }
}
