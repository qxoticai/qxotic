package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.RandomAccessFile;
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
import java.util.Set;
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

    /**
     * The terminal's live region, docker-pull shaped: one row per transfer, repainted in place as
     * ONE write, finished rows freezing into the scrollback as permanent \u2713 lines, messages
     * printed ABOVE the region ({@link #announce}). Its invariants, each one load-bearing:
     *
     * <ul>
     *   <li>ONE writer: every terminal touch happens under this class's lock, as a single print.
     *   <li>A row NEVER wraps: a wrapped row occupies two physical lines and the cursor-up
     *       arithmetic goes off by one, degenerating the region into a scroll-wall. The name column
     *       shrinks first (numbers matter more than file names), then a hard clip guarantees the
     *       invariant on any width.
     *   <li>Geometry never moves: the name column is STICKY while the region lives (a column that
     *       shrank on prune would misalign frozen rows against live ones), and every row field is
     *       fixed-width, so nothing shifts between frames but digits and the bar edge.
     *   <li>Only terminal-bound rows register: a CI log never sees an escape code.
     * </ul>
     */
    static final class Board {
        private static final List<Progress> rows = new ArrayList<>();
        private static final int PREFIX = 3; // " " + spinner + " ", before the name column
        private static int painted; // lines the live region occupies on screen right now
        private static int stickyLabel; // the name column, monotonic while the region lives
        private static int columns = -1;
        private static long lastPaint;
        private static Thread ticker;

        /** A transfer joins the region; the ticker keeps spinners turning through stalls. */
        static synchronized void add(Progress row) {
            rows.add(row);
            if (ticker == null) {
                ticker = new Thread(Board::tick, "jinfer-progress");
                ticker.setDaemon(true);
                ticker.start();
            }
            paint(true);
        }

        /** Data and clock events funnel here; the throttle is shared, ~10 fps total. */
        static synchronized void repaint(boolean force) {
            paint(force);
        }

        /** A message above the region - the "download <ref>" headers, retry notes. */
        static synchronized void announce(String message) {
            StringBuilder out = new StringBuilder();
            if (painted > 0) {
                out.append("\u001b[").append(painted).append("A\r\u001b[J");
                painted = 0;
            }
            out.append(message).append('\n');
            progress().print(out);
            progress().flush();
            paint(true);
        }

        private static void tick() {
            while (true) {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    return;
                }
                synchronized (Board.class) {
                    if (!rows.isEmpty()) {
                        paint(false);
                    }
                }
            }
        }

        private static void paint(boolean force) {
            long now = System.nanoTime();
            if (!force && now - lastPaint < Progress.TICK_NANOS) {
                return;
            }
            lastPaint = now;
            int cols = columns();
            String[] bodies = new String[rows.size()];
            int labelWidth = stickyLabel, body = 0;
            for (int i = 0; i < rows.size(); i++) {
                Progress row = rows.get(i);
                bodies[i] = row.body(now);
                labelWidth = Math.max(labelWidth, row.label.length());
                body = Math.max(body, bodies[i].length());
            }
            labelWidth = Math.max(8, Math.min(labelWidth, cols - 1 - PREFIX - body));
            stickyLabel = labelWidth;
            StringBuilder out = new StringBuilder();
            if (painted > 0) {
                out.append("\u001b[").append(painted).append('A');
            }
            for (int i = 0; i < rows.size(); i++) {
                String line = rows.get(i).renderRow(labelWidth, bodies[i]);
                if (line.length() > cols - 1) {
                    line = line.substring(0, cols - 1); // the never-wrap guarantee
                }
                out.append("\r\u001b[2K").append(line).append('\n');
            }
            painted = rows.size();
            progress().print(out);
            progress().flush();
            // rows finished at the TOP leave the region: the line just painted above the
            // remainder stays in the scrollback as the permanent record
            while (!rows.isEmpty() && rows.get(0).done) {
                rows.remove(0);
                painted--;
            }
            if (rows.isEmpty()) {
                stickyLabel = 0; // the next batch sizes its own column
            }
        }

        /** {@code COLUMNS} > {@code stty size} (asked once) > 80, floored at 40. */
        private static int columns() {
            if (columns > 0) {
                return columns;
            }
            int c = 0;
            String env = System.getenv("COLUMNS");
            if (env != null) {
                try {
                    c = Integer.parseInt(env.strip());
                } catch (NumberFormatException ignored) {
                    // an unparseable width is not worth failing a repaint over
                }
            }
            if (c <= 0) {
                try {
                    Process stty =
                            new ProcessBuilder("stty", "size")
                                    .redirectInput(ProcessBuilder.Redirect.INHERIT)
                                    .start();
                    String[] parts =
                            new String(stty.getInputStream().readAllBytes(), StandardCharsets.UTF_8)
                                    .strip()
                                    .split("\\s+");
                    stty.waitFor();
                    if (parts.length == 2) {
                        c = Integer.parseInt(parts[1]);
                    }
                } catch (Exception unknowable) {
                    // no stty (Windows), no tty on stdin - the 80-column default is the answer
                }
            }
            columns = c >= 40 ? c : 80;
            return columns;
        }
    }

    /** A line above the live region when one is painted, a plain line otherwise. */
    static void announce(String message) {
        Board.announce(message);
    }

    /**
     * Notices a stream whose bytes stopped MOVING and closes it. Downloads carry no request timeout
     * on purpose - a fixed timeout kills legitimately slow links - but a peer that vanishes without
     * a RST would otherwise hold a read parked until the kernel's TCP timeout, which is minutes.
     * Zero bytes for {@code jinfer.downloadStallSeconds} (default 60) closes the stream; that
     * surfaces as an IOException to the retry machinery, and a retry RESUMES.
     */
    private static final class Stall {
        private static final Set<Stall> WATCHED = ConcurrentHashMap.newKeySet();
        private static Thread scanner;

        private final InputStream stream;
        private final AtomicLong bytes = new AtomicLong();
        private long seenBytes = -1; // scanner-local bookkeeping
        private long seenNanos;

        Stall(InputStream stream) {
            this.stream = stream;
            synchronized (Stall.class) {
                if (scanner == null) {
                    scanner = new Thread(Stall::scan, "jinfer-stall-guard");
                    scanner.setDaemon(true);
                    scanner.start();
                }
            }
            WATCHED.add(this);
        }

        void advance(int n) {
            bytes.addAndGet(n);
        }

        void done() {
            WATCHED.remove(this);
        }

        private static void scan() {
            while (true) {
                long limitNanos = Long.getLong("jinfer.downloadStallSeconds", 60) * 1_000_000_000L;
                try {
                    Thread.sleep(Math.max(500, limitNanos / 4_000_000L));
                } catch (InterruptedException e) {
                    return;
                }
                long now = System.nanoTime();
                for (Stall stall : WATCHED) {
                    long moved = stall.bytes.get();
                    if (moved != stall.seenBytes) {
                        stall.seenBytes = moved;
                        stall.seenNanos = now;
                    } else if (stall.seenNanos > 0 && now - stall.seenNanos > limitNanos) {
                        WATCHED.remove(stall);
                        try {
                            stall.stream.close(); // the parked read surfaces as an IOException
                        } catch (IOException ignored) {
                            // the stream is dying either way; the retry owns what happens next
                        }
                    }
                }
            }
        }
    }

    /** {@code downloadThreads=1} plainly means ONE connection at a time - files included. */
    static boolean oneAtATime() {
        return threads() <= 1;
    }

    private static int threads() {
        // -Djinfer.downloadThreads > JINFER_DOWNLOAD_THREADS > the default: the house precedence
        // (property > env > built-in), same as jinfer.models and jinfer.offline
        String configured = System.getProperty("jinfer.downloadThreads");
        if (configured == null || configured.isBlank()) {
            configured = System.getenv("JINFER_DOWNLOAD_THREADS");
        }
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
            int chunks = chunkCount(size);
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
                // NEVER readAllBytes here: a server that ignores Range answers 200 with the
                // WHOLE file, and draining it would stream gigabytes into nothing
                drain.readNBytes(8 * 1024);
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
            // capped: a listing is megabytes at the wildest; an endless body must not be our OOM
            byte[] body = in.readNBytes(64 << 20);
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
        download(url, dest, dest.getFileName().toString(), expectedSize, sha256, headers);
    }

    /**
     * As above with an explicit progress label - for destinations whose file name is NOT the human
     * name (the shared hub cache's content-addressed {@code blobs/<sha256>}).
     */
    static void download(
            String url,
            Path dest,
            String label,
            long expectedSize,
            String sha256,
            Map<String, String> headers)
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
                transfer(url, dest, label, expectedSize, sha256, headers);
            }
        } finally {
            local.unlock();
        }
    }

    private static void transfer(
            String url,
            Path dest,
            String label,
            long expectedSize,
            String sha256,
            Map<String, String> headers)
            throws IOException {
        Path part = sibling(dest, ".part");
        Progress progress = new Progress(label, expectedSize);
        IOException last = null;
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            try {
                if (expectedSize >= PARALLEL_FLOOR) {
                    parallel(url, part, expectedSize, headers, progress);
                    verify(part, sha256, progress);
                } else {
                    sequential(url, part, expectedSize, sha256, headers, progress);
                }
                progress.finish();
                Files.move(part, dest, StandardCopyOption.ATOMIC_MOVE);
                Files.deleteIfExists(sibling(part, ".map"));
                return;
            } catch (IOException e) {
                if (diskFull(e)) {
                    // retrying cannot conjure disk space; say the remedy instead of stalling
                    throw new IOException(
                            label
                                    + ": "
                                    + e.getMessage()
                                    + " - free some space, or point JINFER_MODELS at another"
                                    + " disk",
                            e);
                }
                last = e;
                // the .part and its chunk map survive on purpose: the next attempt resumes
                if (attempt < MAX_ATTEMPTS) {
                    progress.note(e + " - retrying (" + attempt + "/" + MAX_ATTEMPTS + ")");
                }
            }
        }
        throw last;
    }

    private static boolean diskFull(IOException e) {
        String message = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT);
        return message.contains("no space left") || message.contains("not enough space");
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
        int chunks = chunkCount(size);
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
        try (RandomAccessFile allocated = new RandomAccessFile(part.toFile(), "rw");
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
            Stall stall = null;
            try (InputStream in = body(send(URI.create(url), ranged, null), url, true)) {
                stall = new Stall(in);
                byte[] buffer = new byte[BUFFER];
                long remaining = length;
                while (remaining > 0) {
                    int n = in.read(buffer, 0, (int) Math.min(buffer.length, remaining));
                    if (n < 0) {
                        throw new IOException("short chunk " + index + " of " + url);
                    }
                    stall.advance(n);
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
            } finally {
                if (stall != null) {
                    stall.done();
                }
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

    private static int chunkCount(long size) {
        return (int) ((size + CHUNK - 1) / CHUNK);
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

    /**
     * Reads the finished file once to check it - the price of writing chunks out of order. The
     * read-back takes SECONDS on a large file, so it reports through the same row: a bar frozen at
     * 100% reads as a hang, and this is precisely when the download is not done yet.
     */
    private static void verify(Path part, String sha256, Progress progress) throws IOException {
        if (sha256 == null) {
            return;
        }
        progress.verifying();
        MessageDigest digest = sha256Digest();
        digestFile(part, digest, progress);
        checkDigest(digest, sha256, part, progress);
    }

    /** Digests {@code file} into {@code digest}, reporting to {@code progress} when given. */
    private static void digestFile(Path file, MessageDigest digest, Progress progress)
            throws IOException {
        try (InputStream in = Files.newInputStream(file)) {
            byte[] buffer = new byte[BUFFER];
            long hashed = 0;
            for (int n; (n = in.read(buffer)) > 0; ) {
                digest.update(buffer, 0, n);
                hashed += n;
                if (progress != null) {
                    progress.at(hashed);
                }
            }
        }
    }

    /**
     * THE verification law, in one place for both paths: a mismatch deletes the partial (and its
     * chunk map - corrupt is beyond resuming), so the retry is a clean download and the final path
     * never receives unverified bytes.
     */
    private static void checkDigest(MessageDigest digest, String expected, Path part, Progress row)
            throws IOException {
        String actual = HexFormat.of().formatHex(digest.digest());
        if (!actual.equalsIgnoreCase(expected)) {
            Files.deleteIfExists(part);
            Files.deleteIfExists(sibling(part, ".map"));
            throw new IOException(
                    row.label + ": sha256 mismatch, expected " + expected + " but got " + actual);
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
            digestFile(part, digest, null);
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
        Stall stall = null;
        try (InputStream in = body(response, url, true);
                OutputStream out =
                        Files.newOutputStream(
                                part,
                                StandardOpenOption.CREATE,
                                StandardOpenOption.WRITE,
                                StandardOpenOption.APPEND)) {
            stall = new Stall(in);
            byte[] buffer = new byte[BUFFER];
            for (int n; (n = in.read(buffer)) > 0; ) {
                stall.advance(n);
                out.write(buffer, 0, n);
                if (digest != null) {
                    digest.update(buffer, 0, n);
                }
                written += n;
                progress.at(written);
            }
        } finally {
            if (stall != null) {
                stall.done();
            }
        }
        if (expectedSize > 0 && written != expectedSize) {
            throw new IOException("expected " + expectedSize + " bytes, got " + written);
        }
        if (digest != null) {
            checkDigest(digest, sha256, part, progress);
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
                drain.readNBytes(64 * 1024); // redirect bodies are small; hostile ones are not
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
            // capped: the interesting part of an error body is its first line, not its size
            throw new HttpStatusException(
                    status, url, new String(in.readNBytes(64 * 1024), StandardCharsets.UTF_8));
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
        // the AMBIENT root, whatever store instance is downloading: the SPI hands a source its
        // destination, not the store's root, and a lock's home is a jinfer-internal detail. A
        // store built on a custom root litters its locks here - the lock set is tiny and shared,
        // which is also what makes cross-instance downloads of one file exclude each other.
        Path locks = ModelStore.ambientRoot().resolve(".locks");
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
     * One transfer's progress. A terminal gets a live row on the {@link Board}; anything else (a
     * pipe, a CI log, a systemd unit) gets one named line per decile, because a log full of escape
     * codes is worse than no progress at all. A non-UTF-8 console falls back to ASCII glyphs;
     * {@code NO_COLOR} or {@code TERM=dumb} falls all the way back to the plain lines.
     *
     * <p>Worker threads write the fields, the Board's ticker reads them - each is independently
     * volatile, so a frame may briefly mix a fresh phase with a stale rate, which costs one tick of
     * one field and never a tear.
     */
    static final class Progress {

        private static final int WIDTH = 28;
        static final long TICK_NANOS = 100_000_000L;

        /** The classic braille ring; ASCII consoles get the four-spoke wheel. */
        private static final String SPIN =
                "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f";

        private static final String SPIN_ASCII = "|/-\\";

        /** The eighth-blocks a sub-cell leading edge is built from, 1/8 first. */
        private static final String EIGHTHS = "\u258f\u258e\u258d\u258c\u258b\u258a\u2589";

        // process constants, probed once: isTerminal(), NOT console() != null - since JDK 22 a
        // Console is handed out even when output is redirected, so the older check can mistake a
        // CI log for a terminal and fill it with escape codes
        private static final boolean BOARD =
                System.console() != null
                        && System.console().isTerminal()
                        && System.getenv("NO_COLOR") == null
                        && !"dumb".equals(System.getenv("TERM"));
        private static final boolean UNICODE = utf8Console();

        final String label;
        private final long total;
        private volatile long startBytes;
        private volatile long startNanos = System.nanoTime();
        private volatile long lastWritten;
        private volatile long downloadRate; // the DOWNLOAD average, frozen at verify time
        private volatile String phase = ""; // "", or the check's name during the read-back
        volatile boolean done; // the Board prunes finished rows from the top of its region
        private int lastDecile = -1;
        private boolean added; // registered on the Board (once, across retries)

        Progress(String label, long total) {
            this.label = label;
            this.total = total;
        }

        /** Called once the resume point is known, so the rate is measured over new bytes only. */
        void start(long already) {
            phase = "";
            startBytes = already;
            startNanos = System.nanoTime();
            if (BOARD) {
                if (!added) {
                    added = true;
                    Board.add(this);
                }
                return;
            }
            progress()
                    .println(
                            "  "
                                    + label
                                    + "  "
                                    + size(total)
                                    + (already > 0 ? ", resuming at " + size(already) : ""));
        }

        void at(long written) {
            lastWritten = written;
            if (BOARD) {
                Board.repaint(false);
                return;
            }
            int decile = total > 0 ? (int) (10 * written / total) : -1;
            if (decile > lastDecile) {
                lastDecile = decile;
                progress().println("  " + label + "  " + render(written, System.nanoTime()));
            }
        }

        /**
         * The row becomes the sha256 read-back: same line, restarted bar, named phase. The rate
         * field freezes at the download's average from here on - hash throughput answers a question
         * nobody asked, and the final line should immortalize the number that matters.
         */
        void verifying() {
            double seconds = (System.nanoTime() - startNanos) / 1e9;
            downloadRate = seconds > 0 ? (long) ((total - startBytes) / seconds) : 0;
            phase = "sha256";
            startBytes = 0;
            startNanos = System.nanoTime();
            lastDecile = -1;
        }

        void finish() {
            done = true;
            if (BOARD) {
                Board.repaint(true); // the \u2713 shows immediately, and the row may scroll away
            }
        }

        void note(String message) {
            if (BOARD) {
                Board.announce("  " + label + ": " + message);
            } else {
                progress().println("  " + label + ": " + message);
            }
        }

        /** This row's numbers half, measured by the Board before it sizes the name column. */
        String body(long now) {
            return render(done ? total : lastWritten, now);
        }

        /** The Board's view of this transfer: spinner, fitted name, numbers. */
        String renderRow(int labelWidth, String body) {
            return " " + spinner(done) + " " + fit(label, labelWidth) + body;
        }

        /**
         * The numbers half of a row. EVERY field is fixed-width and right-aligned - "36.1 MB" grows
         * into "104 MB", "97.1 MB/s" into "1.5 GB/s" - and the tail is ONE constant-width slot
         * holding "eta 34s" or "sha256": variable widths made the whole line breathe on each
         * repaint, and a slot that resized at the phase flip froze rows with different layouts.
         */
        private String render(long written, long now) {
            double seconds = (now - startNanos) / 1e9;
            long rate =
                    phase.isEmpty()
                            ? (seconds > 0 ? (long) ((written - startBytes) / seconds) : 0)
                            : downloadRate;
            StringBuilder line = new StringBuilder(" ");
            if (total > 0) {
                line.append(UNICODE ? '\u2595' : '[');
                line.append(bar(written, total, WIDTH, UNICODE));
                line.append(UNICODE ? '\u258f' : ']');
                line.append(String.format(Locale.ROOT, " %3d%%", 100 * written / total));
                line.append(String.format(Locale.ROOT, "  %7s/%s", size(written), size(total)));
            } else {
                line.append(label).append(String.format(Locale.ROOT, "  %7s", size(written)));
            }
            line.append(String.format(Locale.ROOT, "  %7s/s", size(rate)));
            String tail;
            if (!phase.isEmpty()) {
                tail = phase;
            } else if (total > 0) {
                tail = "eta " + (rate > 0 && written < total ? eta((total - written) / rate) : "-");
            } else {
                tail = "";
            }
            return line.append(String.format(Locale.ROOT, "  %-10s", tail)).toString();
        }

        /** {@code text} in exactly {@code width} cells: padded, or middle-truncated. */
        private static String fit(String text, int width) {
            if (text.length() <= width) {
                return text + " ".repeat(width - text.length());
            }
            int head = (width - 1) / 2, tail = width - 1 - head;
            return text.substring(0, head)
                    + (UNICODE ? "\u2026" : "~")
                    + text.substring(text.length() - tail);
        }

        /** The live-ness mark: clock-driven, so it turns even when no byte arrives. */
        private static String spinner(boolean done) {
            if (done) {
                return UNICODE ? "\u2713" : "*";
            }
            String frames = UNICODE ? SPIN : SPIN_ASCII;
            return String.valueOf(
                    frames.charAt((int) ((System.nanoTime() / TICK_NANOS) % frames.length())));
        }

        /**
         * The bar cells, {@code width} wide, worn between thin caps by the caller: full blocks, ONE
         * partial block at the leading edge (eighth-cell resolution, so the bar glides instead of
         * chunking), then a SILENT track - the caps give the bar its shape, so the empty side needs
         * no texture. ASCII consoles keep whole cells over a dashed track - there are no partial
         * ASCII blocks worth pretending with.
         */
        static String bar(long written, long total, int width, boolean unicode) {
            long eighths = 8L * width * Math.min(written, total) / total;
            int full = (int) (eighths / 8), part = (int) (eighths % 8);
            StringBuilder bar = new StringBuilder(width);
            bar.append(String.valueOf(unicode ? '\u2588' : '#').repeat(full));
            if (unicode && part > 0) {
                bar.append(EIGHTHS.charAt(part - 1));
            }
            bar.append(String.valueOf(unicode ? ' ' : '-').repeat(width - bar.length()));
            return bar.toString();
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
