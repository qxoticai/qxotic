package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * One file, off the network, onto disk, correctly and fast.
 *
 * <p>FAST is parallel ranged GETs. A single HTTP stream leaves most of a fast link idle, which is
 * why every serious model downloader is multi-connection (HuggingFace's own transfer layer defaults
 * to 16 concurrent range reads). Both hosts here answer {@code Range} with 206 from their CDN, so a
 * large file is cut into chunks, fetched concurrently, and written positionally into a
 * pre-allocated file.
 *
 * <p>CORRECT is four separate hazards, each of which someone WILL hit on a download that runs for
 * minutes and moves tens of gigabytes:
 *
 * <ul>
 *   <li>interruption - chunks completed so far are recorded beside the file, so a resume refetches
 *       only what is missing, out of order and across processes.
 *   <li>corruption - the whole file is verified against the repository's sha256 before it is
 *       published. A silently truncated GGUF memory-maps as a model and fails somewhere far away.
 *   <li>partial visibility - the download lands under a sibling name and is renamed into place, so
 *       the final path either does not exist or is complete. Same directory, because a rename is
 *       only atomic within one filesystem.
 *   <li>concurrency - a lock file makes a second jinfer wait rather than interleave writes, and an
 *       in-JVM lock does the same for two threads, which a file lock alone would reject.
 * </ul>
 *
 * <p>Redirects are followed by hand. Both hosts answer with a 302 to a signed CDN URL, and a signed
 * URL that also carries an {@code Authorization} header is rejected by some CDNs, so the credential
 * is dropped the moment the host changes - the only place it was ever needed.
 */
final class Fetch {

    private Fetch() {}

    /**
     * HTTP/1.1 on purpose: HTTP/2 would multiplex every ranged GET onto ONE connection, and it is
     * the connections, not the requests, that the parallel path exists to multiply.
     */
    private static final HttpClient HTTP =
            HttpClient.newBuilder()
                    .version(HttpClient.Version.HTTP_1_1)
                    .followRedirects(HttpClient.Redirect.NEVER) // see the class note
                    .proxy(Proxies.selector())
                    .connectTimeout(Duration.ofSeconds(30))
                    .build();

    private static final int MAX_REDIRECTS = 5;
    private static final int MAX_ATTEMPTS = 3;
    private static final int BUFFER = 1 << 20;

    /** 32 MB: big enough that per-request overhead vanishes, small enough to resume cheaply. */
    private static final long CHUNK = 32L << 20;

    /** Below two chunks, one stream is faster than the machinery to parallelize it. */
    private static final long PARALLEL_FLOOR = 2 * CHUNK;

    private static final int THREADS = threads();

    /** A metadata call must not hang a startup; a download has no business timing out at all. */
    private static final Duration LISTING_TIMEOUT = Duration.ofSeconds(20);

    /** Two threads of one JVM cannot both hold a {@link FileLock}, so they queue here first. */
    private static final Map<Path, ReentrantLock> IN_PROCESS = new ConcurrentHashMap<>();

    private static int threads() {
        String configured = System.getenv("JINFER_DOWNLOAD_THREADS");
        if (configured != null && !configured.isBlank()) {
            try {
                return Math.max(1, Integer.parseInt(configured.strip()));
            } catch (NumberFormatException ignored) {
                // an unparseable override is not worth failing a download over
            }
        }
        return Math.clamp(Runtime.getRuntime().availableProcessors(), 4, 8);
    }

    /**
     * How many bytes a download of {@code size} still has to move, given whatever a previous
     * attempt left beside {@code dest}. The disk check needs this rather than the full size: the
     * machine that ran out of space at 30 GB of 40 is exactly the machine that resumes, and
     * refusing it for want of the 30 GB it already has would make resume useless where it matters
     * most.
     */
    static long remainingBytes(Path dest, long size) {
        try {
            Path part = sibling(dest, ".part");
            if (!Files.exists(part)) {
                return size;
            }
            Path mapFile = sibling(part, ".map");
            int chunks = (int) ((size + CHUNK - 1) / CHUNK);
            if (Files.exists(mapFile)
                    && Files.size(mapFile) == chunks
                    && Files.size(part) == size) {
                byte[] done = Files.readAllBytes(mapFile);
                long missing = 0;
                for (int i = 0; i < chunks; i++) {
                    if (done[i] == 0) {
                        missing += chunkSize(i, chunks, size); // the last chunk is a short one
                    }
                }
                return missing; // the pre-allocated .part is sparse until its chunks are written
            }
            return Math.max(0, size - Files.size(part)); // the sequential path appends
        } catch (IOException unreadable) {
            return size; // cannot tell: assume the worst and let the check be conservative
        }
    }

    /**
     * The size of {@code url}, or -1 when the server will not say. A one-byte ranged GET rather
     * than a HEAD: every host that serves Range answers it, and some that redirect to a CDN handle
     * HEAD badly. Knowing the size up front is what lets a plain URL use the parallel path and the
     * disk check, neither of which can work from a stream of unknown length.
     */
    static long sizeOf(String url, Map<String, String> headers) {
        Map<String, String> ranged = new LinkedHashMap<>(headers);
        ranged.put("Range", "bytes=0-0");
        try {
            HttpResponse<InputStream> response = send(URI.create(url), ranged, LISTING_TIMEOUT);
            try (InputStream drain = response.body()) {
                drain.readAllBytes();
            }
            String contentRange = response.headers().firstValue("content-range").orElse("");
            int slash = contentRange.lastIndexOf('/');
            if (response.statusCode() == 206 && slash > 0) {
                return Long.parseLong(contentRange.substring(slash + 1).strip());
            }
            return response.statusCode() == 200
                    ? response.headers().firstValueAsLong("content-length").orElse(-1)
                    : -1;
        } catch (IOException | NumberFormatException unknown) {
            return -1; // a size we cannot learn is not a reason to refuse the download
        }
    }

    /** A GET whose body is a String - the listing APIs. */
    static String getString(String url, Map<String, String> headers) throws IOException {
        HttpResponse<InputStream> response = send(URI.create(url), headers, LISTING_TIMEOUT);
        try (InputStream in = response.body()) {
            byte[] body = in.readAllBytes();
            if (response.statusCode() != 200) {
                throw new HttpStatusException(
                        response.statusCode(), url, new String(body, StandardCharsets.UTF_8));
            }
            return new String(body, StandardCharsets.UTF_8);
        }
    }

    /**
     * Downloads {@code url} to {@code dest}, resuming what a previous attempt left and verifying
     * {@code sha256} when the repository supplied one. Returns with {@code dest} either complete or
     * absent. Blocks while another process downloads the same file, then returns its result.
     */
    static void download(
            String url, Path dest, long expectedSize, String sha256, Map<String, String> headers)
            throws IOException {
        Files.createDirectories(dest.getParent());
        ReentrantLock local =
                IN_PROCESS.computeIfAbsent(dest.toAbsolutePath(), p -> new ReentrantLock());
        local.lock();
        try {
            Path lockFile = lockFileFor(dest);
            // NOT delete-on-close: on Windows a pending delete makes another process's open of the
            // same lock file fail outright, turning "wait your turn" into "crash"
            try (FileChannel channel =
                            FileChannel.open(
                                    lockFile, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                    FileLock fileLock = channel.lock()) {
                if (Files.exists(dest)) {
                    return; // whoever held the lock finished it
                }
                transfer(url, dest, expectedSize, sha256, headers);
            }
        } finally {
            local.unlock();
        }
    }

    private static void transfer(
            String url, Path dest, long expectedSize, String sha256, Map<String, String> headers)
            throws IOException {
        Path part = sibling(dest, ".part");
        Progress progress = new Progress(dest.getFileName().toString(), expectedSize);
        IOException last = null;
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            try {
                if (expectedSize >= PARALLEL_FLOOR) {
                    parallel(url, part, expectedSize, headers, progress);
                    verify(part, sha256, dest.getFileName().toString());
                } else {
                    sequential(url, part, expectedSize, sha256, headers, progress);
                }
                progress.finish();
                Files.move(part, dest, StandardCopyOption.ATOMIC_MOVE);
                Files.deleteIfExists(sibling(part, ".map"));
                return;
            } catch (IOException e) {
                last = e;
                // the .part and its chunk map survive on purpose: the next attempt resumes
                if (attempt < MAX_ATTEMPTS) {
                    progress.note(e + " - retrying (" + attempt + "/" + MAX_ATTEMPTS + ")");
                }
            }
        }
        throw last;
    }

    // ---- parallel path ----

    /**
     * Splits the file into chunks and fetches the ones the map says are missing. Workers write
     * positionally into the pre-allocated {@code part}, which is safe on every platform and needs
     * no coordination between them beyond the map.
     */
    private static void parallel(
            String url, Path part, long size, Map<String, String> headers, Progress progress)
            throws IOException {
        int chunks = (int) ((size + CHUNK - 1) / CHUNK);
        Path mapFile = sibling(part, ".map");
        byte[] done = chunkMap(mapFile, chunks, part, size);
        long already = 0;
        List<Integer> todo = new ArrayList<>();
        for (int i = 0; i < chunks; i++) {
            if (done[i] != 0) {
                already += chunkSize(i, chunks, size);
            } else {
                todo.add(i);
            }
        }
        progress.start(already);
        if (todo.isEmpty()) {
            return;
        }
        AtomicLong written = new AtomicLong(already);
        // setLength, NOT FileChannel.truncate: truncate only ever SHRINKS, so it would leave the
        // file as short as its highest written byte. The chunk map is only trusted when the file
        // beside it is already full length, so without a real pre-allocation every resume would
        // throw the map away and start over.
        try (java.io.RandomAccessFile allocated =
                        new java.io.RandomAccessFile(part.toFile(), "rw");
                FileChannel map =
                        FileChannel.open(
                                mapFile, StandardOpenOption.WRITE, StandardOpenOption.CREATE);
                ExecutorService pool =
                        Executors.newFixedThreadPool(Math.min(THREADS, todo.size()))) {
            allocated.setLength(size);
            FileChannel file = allocated.getChannel();
            List<Future<?>> futures = new ArrayList<>(todo.size());
            for (int index : todo) {
                futures.add(
                        pool.submit(
                                () -> {
                                    chunk(
                                            url, headers, file, index, chunks, size, written,
                                            progress);
                                    // the data MUST reach the disk before the map says it did:
                                    // a crash that persisted the bit but not the bytes would
                                    // resume over a hole, and the hole only surfaces as a sha256
                                    // mismatch after everything else has been downloaded again
                                    file.force(false);
                                    map.write(ByteBuffer.wrap(new byte[] {1}), index);
                                    return null;
                                }));
            }
            progress.pump(written, futures);
            for (Future<?> future : futures) {
                await(future);
            }
            map.force(true);
        }
    }

    /** One chunk, retried on its own: a single flaky range must not restart the whole file. */
    private static void chunk(
            String url,
            Map<String, String> headers,
            FileChannel file,
            int index,
            int chunks,
            long size,
            AtomicLong written,
            Progress progress)
            throws IOException {
        long start = index * CHUNK;
        long length = chunkSize(index, chunks, size);
        IOException last = null;
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            Map<String, String> ranged = new LinkedHashMap<>(headers);
            ranged.put("Range", "bytes=" + start + "-" + (start + length - 1));
            long at = start;
            try (InputStream in = body(send(URI.create(url), ranged, null), url, true)) {
                byte[] buffer = new byte[BUFFER];
                long remaining = length;
                while (remaining > 0) {
                    int n = in.read(buffer, 0, (int) Math.min(buffer.length, remaining));
                    if (n < 0) {
                        throw new IOException("short chunk " + index + " of " + url);
                    }
                    ByteBuffer slice = ByteBuffer.wrap(buffer, 0, n);
                    while (slice.hasRemaining()) {
                        at += file.write(slice, at);
                    }
                    remaining -= n;
                    progress.at(written.addAndGet(n));
                }
                return;
            } catch (IOException e) {
                last = e;
                written.addAndGet(start - at); // un-count what this failed attempt had reported
            }
        }
        throw last;
    }

    /**
     * The chunk map: one byte per chunk, 1 when that chunk is on disk. Discarded whenever it cannot
     * describe the file beside it, because a stale map would publish a file with holes in it.
     */
    private static byte[] chunkMap(Path mapFile, int chunks, Path part, long size)
            throws IOException {
        boolean usable =
                Files.exists(mapFile)
                        && Files.size(mapFile) == chunks
                        && Files.exists(part)
                        && Files.size(part) == size;
        if (!usable) {
            Files.deleteIfExists(mapFile);
            Files.write(mapFile, new byte[chunks]);
            return new byte[chunks];
        }
        return Files.readAllBytes(mapFile);
    }

    private static long chunkSize(int index, int chunks, long size) {
        return index == chunks - 1 ? size - index * CHUNK : CHUNK;
    }

    private static void await(Future<?> future) throws IOException {
        try {
            future.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while downloading", e);
        } catch (ExecutionException e) {
            switch (e.getCause()) {
                case IOException io -> throw io;
                case RuntimeException runtime -> throw runtime;
                case Error error -> throw error;
                case Throwable other -> throw new IOException(other);
            }
        }
    }

    /** Reads the finished file once to check it. The price of writing chunks out of order. */
    private static void verify(Path part, String sha256, String label) throws IOException {
        if (sha256 == null) {
            return;
        }
        MessageDigest digest = sha256Digest();
        try (InputStream in = Files.newInputStream(part)) {
            byte[] buffer = new byte[BUFFER];
            for (int n; (n = in.read(buffer)) > 0; ) {
                digest.update(buffer, 0, n);
            }
        }
        String actual = HexFormat.of().formatHex(digest.digest());
        if (!actual.equalsIgnoreCase(sha256)) {
            // corrupt beyond resuming: drop both, so the retry is a clean download
            Files.deleteIfExists(part);
            Files.deleteIfExists(sibling(part, ".map"));
            throw new IOException(
                    label + ": sha256 mismatch, expected " + sha256 + " but got " + actual);
        }
    }

    // ---- sequential path (small files, or a size the host would not state) ----

    private static void sequential(
            String url,
            Path part,
            long expectedSize,
            String sha256,
            Map<String, String> headers,
            Progress progress)
            throws IOException {
        long have = Files.exists(part) ? Files.size(part) : 0;
        if (expectedSize > 0 && have > expectedSize) {
            Files.delete(part); // a .part longer than the file it claims to be is not a resume
            have = 0;
        }
        MessageDigest digest = sha256 == null ? null : sha256Digest();
        if (have > 0 && digest != null) {
            // a digest cannot be resumed across processes: re-read what is already on disk. One
            // disk pass against a network download is a trade worth making for a real checksum.
            try (InputStream in = Files.newInputStream(part)) {
                byte[] buffer = new byte[BUFFER];
                for (int n; (n = in.read(buffer)) > 0; ) {
                    digest.update(buffer, 0, n);
                }
            }
        }
        Map<String, String> ranged = headers;
        if (have > 0) {
            ranged = new LinkedHashMap<>(headers);
            ranged.put("Range", "bytes=" + have + "-");
        }
        HttpResponse<InputStream> response = send(URI.create(url), ranged, null);
        if (have > 0 && response.statusCode() == 200) {
            // the server ignored the range: start over rather than append to a stale prefix
            have = 0;
            digest = sha256 == null ? null : sha256Digest();
            Files.deleteIfExists(part);
        }
        progress.start(have);
        long written = have;
        try (InputStream in = body(response, url, true);
                OutputStream out =
                        Files.newOutputStream(
                                part,
                                StandardOpenOption.CREATE,
                                StandardOpenOption.WRITE,
                                StandardOpenOption.APPEND)) {
            byte[] buffer = new byte[BUFFER];
            for (int n; (n = in.read(buffer)) > 0; ) {
                out.write(buffer, 0, n);
                if (digest != null) {
                    digest.update(buffer, 0, n);
                }
                written += n;
                progress.at(written);
            }
        }
        if (expectedSize > 0 && written != expectedSize) {
            throw new IOException("expected " + expectedSize + " bytes, got " + written);
        }
        if (digest != null) {
            String actual = HexFormat.of().formatHex(digest.digest());
            if (!actual.equalsIgnoreCase(sha256)) {
                Files.deleteIfExists(part); // corrupt: never leave it to be resumed
                throw new IOException("sha256 mismatch, expected " + sha256 + " but got " + actual);
            }
        }
    }

    // ---- HTTP ----

    /** Sends a GET, following redirects by hand and dropping credentials off-host. */
    private static HttpResponse<InputStream> send(
            URI uri, Map<String, String> headers, Duration timeout) throws IOException {
        URI current = uri;
        Map<String, String> currentHeaders = headers;
        for (int redirect = 0; ; redirect++) {
            HttpRequest.Builder request = HttpRequest.newBuilder(current).GET();
            currentHeaders.forEach(request::header);
            if (timeout != null) {
                request.timeout(timeout);
            }
            HttpResponse<InputStream> response;
            try {
                response = HTTP.send(request.build(), HttpResponse.BodyHandlers.ofInputStream());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted while fetching " + current, e);
            }
            int status = response.statusCode();
            if (status < 300 || status >= 400) {
                return response;
            }
            String location = response.headers().firstValue("location").orElse(null);
            try (InputStream drain = response.body()) {
                drain.readAllBytes();
            }
            if (location == null || redirect == MAX_REDIRECTS) {
                throw new IOException("redirect loop or missing Location for " + uri);
            }
            URI next = current.resolve(location);
            if (!next.getHost().equalsIgnoreCase(current.getHost())) {
                // the CDN URL is already signed; an Authorization header alongside it is both
                // useless and, on some CDNs, a 400
                currentHeaders = new LinkedHashMap<>(currentHeaders);
                currentHeaders.remove("Authorization");
            }
            current = next;
        }
    }

    /** The response body, or the response's own explanation of why there is not one. */
    private static InputStream body(HttpResponse<InputStream> response, String url, boolean ranged)
            throws IOException {
        int status = response.statusCode();
        if (status == 200 || (ranged && status == 206)) {
            return response.body();
        }
        try (InputStream in = response.body()) {
            throw new HttpStatusException(
                    status, url, new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    private static Path sibling(Path path, String suffix) {
        return path.resolveSibling(path.getFileName() + suffix);
    }

    /**
     * The lock for {@code dest}, in the cache root's own {@code .locks} folder rather than beside
     * the model.
     *
     * <p>Two reasons. A lock file must NEVER be deleted - unlinking one another process is already
     * blocked on lets a third process create a new file and lock that instead, so both would think
     * they hold it - which means locks are permanent litter, and permanent litter does not belong
     * in a directory whose whole promise is that it contains models and nothing else. And keeping
     * them out of the model directory is what lets {@code rm -rf} of a repository stay safe on
     * every platform, including Windows, where an open lock file cannot be deleted at all.
     */
    private static Path lockFileFor(Path dest) throws IOException {
        Path locks = ModelStore.root().resolve(".locks");
        Files.createDirectories(locks);
        String key =
                HexFormat.of()
                        .formatHex(
                                sha256Digest()
                                        .digest(
                                                dest.toAbsolutePath()
                                                        .toString()
                                                        .getBytes(StandardCharsets.UTF_8)))
                        .substring(0, 32);
        return locks.resolve(key + ".lock");
    }

    private static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** A non-2xx response, carrying enough to explain itself. */
    static final class HttpStatusException extends IOException {

        final int status;

        HttpStatusException(int status, String url, String body) {
            super("HTTP " + status + " for " + url + (body.isBlank() ? "" : ": " + brief(body)));
            this.status = status;
        }

        private static String brief(String body) {
            String trimmed = body.strip().replaceAll("\\s+", " ");
            return trimmed.length() > 200 ? trimmed.substring(0, 200) + "..." : trimmed;
        }
    }

    static PrintStream progress() {
        return System.err;
    }

    /**
     * One line of download progress. A terminal gets a bar rewritten in place; anything else (a
     * pipe, a CI log, a systemd unit) gets one line per decile, because a log full of carriage
     * returns is worse than no progress at all. {@code NO_COLOR} and a non-UTF-8 console both fall
     * back to ASCII, so the bar renders on a Windows console too.
     */
    static final class Progress {

        private static final int WIDTH = 28;
        private static final long TICK_NANOS = 100_000_000L;

        private final String label;
        private final long total;
        // isTerminal(), NOT console() != null: since JDK 22 a Console is handed out even when
        // output is redirected, so the older check can mistake a CI log or a pipe for a terminal
        // and fill it with carriage returns
        private final boolean tty =
                System.console() != null
                        && System.console().isTerminal()
                        && System.getenv("NO_COLOR") == null;
        private final boolean unicode = utf8Console();
        private long startBytes;
        private long startNanos = System.nanoTime();
        private long lastPrint;
        private int lastDecile = -1;

        Progress(String label, long total) {
            this.label = label;
            this.total = total;
        }

        /** Called once the resume point is known, so the rate is measured over new bytes only. */
        void start(long already) {
            startBytes = already;
            startNanos = System.nanoTime();
            if (!tty) {
                progress()
                        .println(
                                "  "
                                        + size(total)
                                        + (already > 0 ? ", resuming at " + size(already) : ""));
            }
        }

        void note(String message) {
            if (tty) {
                progress().println();
            }
            progress().println("  " + message);
        }

        /** Renders while {@code futures} run - the parallel path has no single stream to hook. */
        void pump(AtomicLong written, List<Future<?>> futures) {
            while (!futures.stream().allMatch(Future::isDone)) {
                at(written.get());
                try {
                    Thread.sleep(Duration.ofMillis(100));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
            at(written.get());
        }

        void at(long written) {
            long now = System.nanoTime();
            if (tty) {
                if (now - lastPrint < TICK_NANOS) {
                    return;
                }
                lastPrint = now;
                progress().print("\r" + render(written, now));
                progress().flush();
                return;
            }
            int decile = total > 0 ? (int) (10 * written / total) : -1;
            if (decile > lastDecile) {
                lastDecile = decile;
                progress().println("  " + render(written, now));
            }
        }

        void finish() {
            if (tty) {
                progress().print("\r" + render(total, System.nanoTime()));
                progress().println();
            }
        }

        private String render(long written, long now) {
            double seconds = (now - startNanos) / 1e9;
            long rate = seconds > 0 ? (long) ((written - startBytes) / seconds) : 0;
            StringBuilder line = new StringBuilder(" ");
            if (total > 0) {
                int filled = (int) (WIDTH * Math.min(written, total) / total);
                line.append(String.valueOf(unicode ? '█' : '#').repeat(filled));
                line.append(String.valueOf(unicode ? '░' : '-').repeat(WIDTH - filled));
                line.append(String.format(Locale.ROOT, " %3d%%", 100 * written / total));
                line.append("  ").append(size(written)).append('/').append(size(total));
            } else {
                line.append(label).append("  ").append(size(written));
            }
            line.append("  ").append(size(rate)).append("/s");
            if (total > 0 && rate > 0 && written < total) {
                line.append("  ").append(eta((total - written) / rate));
            }
            return line.toString();
        }

        private static boolean utf8Console() {
            Charset charset =
                    System.console() != null
                            ? System.console().charset()
                            : Charset.defaultCharset();
            return charset.contains(StandardCharsets.UTF_8);
        }

        private static String eta(long seconds) {
            if (seconds < 60) {
                return seconds + "s";
            }
            if (seconds < 3600) {
                return seconds / 60 + "m" + seconds % 60 + "s";
            }
            return seconds / 3600 + "h" + (seconds % 3600) / 60 + "m";
        }
    }

    /** Human bytes, three significant digits, the units a download speed is quoted in. */
    static String size(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        String[] units = {"KB", "MB", "GB", "TB"};
        double value = bytes;
        int unit = -1;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit++;
        }
        return String.format(Locale.ROOT, value >= 100 ? "%.0f %s" : "%.1f %s", value, units[unit]);
    }
}
