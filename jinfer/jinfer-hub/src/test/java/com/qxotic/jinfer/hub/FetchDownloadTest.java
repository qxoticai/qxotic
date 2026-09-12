package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The download's own guarantees against a {@link FileServer}: sha256 enforcement, the resume
 * contract, and what is (not) left behind on failure. These are the bytes-on-disk properties every
 * {@link ModelSource} built on {@code Fetch.download} inherits.
 */
class FetchDownloadTest {

    private static final String PAYLOAD =
            "weights, but long enough to have a middle worth resuming from";

    @AfterEach
    void pathLocksAreReleased() throws ReflectiveOperationException {
        Field registry = Fetch.class.getDeclaredField("LOCK_REGISTRY");
        registry.setAccessible(true);
        assertTrue(((Map<?, ?>) registry.get(null)).isEmpty());
    }

    private static String sha256(String value) {
        return sha256(value.getBytes(StandardCharsets.UTF_8));
    }

    private static String sha256(byte[] value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    @Test
    void redirectsStripAuthorizationOnPortChange() throws IOException {
        try (FileServer target = FileServer.start().serve("/target", "ok");
                FileServer source = FileServer.start().redirect("/source", target.url("/target"))) {
            assertEquals(
                    "ok",
                    Fetch.getString(
                            source.url("/source"), Map.of("authorization", "Bearer audit-dummy")));
            assertEquals("Bearer audit-dummy", source.lastHeader("/source", "Authorization"));
            assertNull(target.lastHeader("/target", "Authorization"));
        }
    }

    @Test
    void redirectsKeepAuthorizationWithinOneOrigin() throws IOException {
        try (FileServer server =
                FileServer.start().redirect("/source", "/target").serve("/target", "ok")) {
            assertEquals(
                    "ok",
                    Fetch.getString(
                            server.url("/source"), Map.of("Authorization", "Bearer audit-dummy")));
            assertEquals("Bearer audit-dummy", server.lastHeader("/target", "Authorization"));
        }
    }

    @Test
    void authorizationStaysStrippedForTheRestOfTheChain() throws IOException {
        try (FileServer first = FileServer.start();
                FileServer second = FileServer.start()) {
            first.redirect("/source", second.url("/middle")).serve("/target", "ok");
            second.redirect("/middle", first.url("/target"));

            assertEquals(
                    "ok",
                    Fetch.getString(
                            first.url("/source"), Map.of("Authorization", "Bearer audit-dummy")));
            assertEquals("Bearer audit-dummy", first.lastHeader("/source", "Authorization"));
            assertNull(second.lastHeader("/middle", "Authorization"));
            assertNull(first.lastHeader("/target", "Authorization"));
        }
    }

    @Test
    void originsIncludeSchemeHostAndEffectivePort() {
        assertTrue(
                Fetch.sameOrigin(
                        URI.create("http://EXAMPLE.com/model"),
                        URI.create("http://example.com:80/other")));
        assertTrue(
                Fetch.sameOrigin(
                        URI.create("https://example.com/model"),
                        URI.create("https://example.com:443/other")));
        assertFalse(
                Fetch.sameOrigin(
                        URI.create("http://example.com/model"),
                        URI.create("https://example.com/model")));
        assertFalse(
                Fetch.sameOrigin(
                        URI.create("http://example.com/model"),
                        URI.create("http://other.example/model")));
        assertFalse(
                Fetch.sameOrigin(
                        URI.create("http://example.com:8080/model"),
                        URI.create("http://example.com:8081/model")));
    }

    @Test
    void unsafeRedirectTargetsAreIoFailures() {
        URI source = URI.create("https://example.com/model");
        assertThrows(
                IOException.class, () -> Fetch.redirectTarget(source, "http://example.com/model"));
        assertThrows(IOException.class, () -> Fetch.redirectTarget(source, "file:///tmp/model"));
        assertThrows(
                IOException.class,
                () -> Fetch.redirectTarget(source, "https://example.com:99999/model"));
        assertThrows(IOException.class, () -> Fetch.redirectTarget(source, "http://["));
    }

    @Test
    void queryOnlyRedirectsKeepTheCurrentPath() throws IOException {
        assertEquals(
                URI.create("https://example.com/a/model?new"),
                Fetch.redirectTarget(
                        URI.create("https://example.com/a/model?old"), "?new#ignored"));
        assertEquals(
                URI.create("https://example.com//other.example/model?new"),
                Fetch.redirectTarget(
                        URI.create("https://example.com//other.example/model?old"), "?new"));
    }

    @Test
    void nonRedirect3xxResponsesAreNotFollowed() throws IOException {
        try (FileServer target = FileServer.start().serve("/target", "wrong");
                FileServer source =
                        FileServer.start().redirect("/source", 304, target.url("/target"))) {
            var failure =
                    assertThrows(
                            Fetch.HttpStatusException.class,
                            () -> Fetch.getString(source.url("/source"), Map.of()));
            assertEquals(304, failure.status);
            assertEquals(0, target.hits("/target"));
        }
    }

    @Test
    void redirectBodiesAreClosedWithoutBeingDrained() throws IOException {
        try (FileServer target = FileServer.start().serve("/target", "ok");
                FileServer source =
                        FileServer.start().stalledRedirect("/source", target.url("/target"))) {
            assertTimeoutPreemptively(
                    Duration.ofSeconds(2),
                    () -> assertEquals("ok", Fetch.getString(source.url("/source"), Map.of())));
        }
    }

    @Test
    void redirectLoopsStopAtTheLimit() throws IOException {
        try (FileServer server = FileServer.start().redirect("/loop", "/loop")) {
            assertThrows(IOException.class, () -> Fetch.getString(server.url("/loop"), Map.of()));
            assertEquals(6, server.hits("/loop"));
        }
    }

    @Test
    void aRefusedFileIsNotRetried(@TempDir Path dir) throws IOException {
        // 401/403/404 are the server's answer, not a bad moment: one request, the status kept
        try (FileServer server = FileServer.start().deny("/gated.gguf", 401)) {
            Path dest = dir.resolve("gated.gguf");

            var failure =
                    assertThrows(
                            Fetch.HttpStatusException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/gated.gguf"), dest, 3, null, Map.of()));

            assertEquals(401, failure.status);
            assertEquals(1, server.hits("/gated.gguf"), "no retry");
        }
    }

    @Test
    void aRefusedChunkedFileIsNotRetriedEither(@TempDir Path dir) throws IOException {
        // the parallel path: every chunk asks once, the first refusal ends the download
        try (FileServer server = FileServer.start().deny("/gated.gguf", 401)) {
            Path dest = dir.resolve("gated.gguf");

            var failure =
                    assertThrows(
                            Fetch.HttpStatusException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/gated.gguf"),
                                            dest,
                                            Fetch.PARALLEL_FLOOR,
                                            null,
                                            Map.of()));

            assertEquals(401, failure.status);
            assertTrue(
                    server.hits("/gated.gguf") <= 2,
                    "at most one request per chunk, got " + server.hits("/gated.gguf"));
        }
    }

    @Test
    void aMatchingSha256IsVerifiedSilently(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals(PAYLOAD, Files.readString(dest));
        }
    }

    @Test
    void aServerIgnoringRangeStillDownloadsCorrectlyAboveTheParallelFloor(@TempDir Path dir)
            throws IOException {
        // every chunk beyond the first used to accept a 200 (the whole file) and write it at its
        // own offset: a corrupt file, published silently when no sha256 was known
        byte[] big = new byte[(int) Fetch.PARALLEL_FLOOR + 1];
        new Random(11).nextBytes(big);
        try (FileServer server =
                FileServer.start().serve("/big.bin", big).ignoringRange("/big.bin")) {
            Path dest = dir.resolve("big.bin");
            Fetch.download(server.url("/big.bin"), dest, big.length, null, Map.of());
            assertArrayEquals(big, Files.readAllBytes(dest));
        }
    }

    @Test
    void aParallelResumeRestartsWhenTheRemoteChanged(@TempDir Path dir) throws IOException {
        // a chunk map trusted on sizes alone kept a stale chunk of a republished file of the
        // same size; the parallel path now sends the first response's validator as If-Range,
        // and a changed remote (200 to a ranged request) restarts the transfer from scratch
        byte[] big = new byte[(int) Fetch.PARALLEL_FLOOR + 1];
        new Random(12).nextBytes(big);
        try (FileServer server =
                FileServer.start().serve("/big.bin", big).etag("/big.bin", "\"v2\"")) {
            Path dest = dir.resolve("big.bin");
            Path part = dest.resolveSibling("big.bin.part");
            byte[] stale = new byte[big.length]; // "chunk 0 done" of the previous file
            Files.write(part, stale);
            Files.write(dest.resolveSibling("big.bin.part.map"), new byte[] {1, 0, 0});
            Files.writeString(dest.resolveSibling("big.bin.part.etag"), "\"v1\"");

            Fetch.download(server.url("/big.bin"), dest, big.length, null, Map.of());

            assertEquals("\"v1\"", server.lastHeader("/big.bin", "If-Range"));
            assertArrayEquals(big, Files.readAllBytes(dest));
            assertTrue(!Files.exists(dest.resolveSibling("big.bin.part.etag")), "cleaned up");
        }
    }

    @Test
    void aPartAlreadyAtFullSizeIsNotResumed(@TempDir Path dir) throws IOException {
        // a crash between the last byte and the rename left a full-size .part; resuming it asked
        // for a range past the end, a 416 on every attempt
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            Files.writeString(dest.resolveSibling("m.gguf.part"), PAYLOAD);

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals(PAYLOAD, Files.readString(dest));
        }
    }

    @Test
    void aResumeCarriesTheFirstResponsesValidator(@TempDir Path dir) throws IOException {
        // If-Range with the validator the first response gave: a remote that changed answers
        // the whole file, never a tail of a different file appended to the stale prefix
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            Path part = dest.resolveSibling("m.gguf.part");
            Files.writeString(part, PAYLOAD.substring(0, PAYLOAD.length() / 2));
            Files.writeString(dest.resolveSibling("m.gguf.part.etag"), "\"v1\"");

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals("\"v1\"", server.lastHeader("/m.gguf", "If-Range"));
            assertEquals(PAYLOAD, Files.readString(dest));
            assertTrue(!Files.exists(dest.resolveSibling("m.gguf.part.etag")), "cleaned up");
        }
    }

    @Test
    void aWrongSha256FailsAndLeavesNothingBehind(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            String wrong = "0".repeat(64);

            var failure =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/m.gguf"),
                                            dest,
                                            PAYLOAD.length(),
                                            wrong,
                                            Map.of()));
            assertTrue(failure.getMessage().contains("sha256 mismatch"), failure.getMessage());
            assertTrue(Files.notExists(dest), "dest is either complete or absent");
            // the law is stronger than that: the partial is DELETED, because resuming bytes that
            // can never match the hash would fail every future attempt the same way
            assertTrue(Files.notExists(dest.resolveSibling("m.gguf.part")));
        }
    }

    @Test
    void aResumeFetchesOnlyWhatIsMissing(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            int kept = PAYLOAD.length() / 2;
            Files.writeString(dest.resolveSibling("m.gguf.part"), PAYLOAD.substring(0, kept));

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals(PAYLOAD, Files.readString(dest));
            assertEquals(
                    "bytes=" + kept + "-",
                    server.lastRange("/m.gguf"),
                    "the fetch continued where the .part ended");
        }
    }

    @Test
    void anUndersizedResponseFailsRatherThanTruncating(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");

            assertThrows(
                    IOException.class,
                    () ->
                            Fetch.download(
                                    server.url("/m.gguf"),
                                    dest,
                                    PAYLOAD.length() + 1000,
                                    null,
                                    Map.of()));
            assertTrue(Files.notExists(dest));
        }
    }

    @Test
    void freshParallelResponsesMustAgreeOnTheirETag(@TempDir Path dir) throws IOException {
        long size = Fetch.PARALLEL_FLOOR;
        byte[] first = new byte[(int) (size / 2)];
        byte[] second = new byte[first.length];
        Arrays.fill(first, (byte) 'A');
        Arrays.fill(second, (byte) 'B');
        CountDownLatch requests = new CountDownLatch(2);
        try (FileServer server = FileServer.start()) {
            server.respond(
                    "/mixed.bin",
                    exchange -> {
                        long start = rangeStart(exchange);
                        requests.countDown();
                        try {
                            if (!requests.await(5, TimeUnit.SECONDS))
                                throw new IOException("missing peer");
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IOException(e);
                        }
                        reply(
                                exchange,
                                "bytes %d-%d/%d".formatted(start, start + first.length - 1, size),
                                start == 0 ? "\"v1\"" : "\"v2\"",
                                start == 0 ? first : second,
                                false);
                    });
            Path dest = dir.resolve("mixed.bin");
            IOException failure =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/mixed.bin"), dest, size, null, Map.of()));
            assertTrue(failure.getMessage().contains("ETag changed"), failure.getMessage());
            assertDiscarded(dest);
        }
    }

    @Test
    void aParallelChunkMustDescribeTheRequestedRange(@TempDir Path dir) throws IOException {
        long size = Fetch.PARALLEL_FLOOR;
        byte[] chunk = new byte[(int) (size / 2)];
        try (FileServer server =
                FileServer.start()
                        .respond(
                                "/wrong.bin",
                                exchange ->
                                        reply(
                                                exchange,
                                                "bytes 0-" + (chunk.length - 1) + "/" + size,
                                                "\"stable\"",
                                                chunk,
                                                false))) {
            Path dest = dir.resolve("wrong.bin");
            IOException failure =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/wrong.bin"), dest, size, null, Map.of()));
            assertTrue(failure.getMessage().contains("Content-Range"), failure.getMessage());
            assertDiscarded(dest);
        }
    }

    @Test
    void aChunkRetryKeepsTheOriginalValidator(@TempDir Path dir) throws IOException {
        long size = Fetch.PARALLEL_FLOOR;
        byte[] chunk = new byte[(int) (size / 2)];
        AtomicInteger attempts = new AtomicInteger();
        AtomicReference<String> retryTag = new AtomicReference<>();
        try (FileServer server = FileServer.start()) {
            server.respond(
                    "/retry.bin",
                    exchange -> {
                        long start = rangeStart(exchange);
                        int attempt = start == 0 ? attempts.incrementAndGet() : 0;
                        if (attempt == 2)
                            retryTag.set(exchange.getRequestHeaders().getFirst("If-Range"));
                        reply(
                                exchange,
                                "bytes %d-%d/%d".formatted(start, start + chunk.length - 1, size),
                                attempt > 1 ? "\"v2\"" : "\"v1\"",
                                attempt == 1 ? new byte[] {1} : chunk,
                                true); // a short first body forces a retry of chunk zero
                    });
            Path dest = dir.resolve("retry.bin");
            IOException failure =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/retry.bin"), dest, size, null, Map.of()));
            assertEquals("\"v1\"", retryTag.get());
            assertTrue(failure.getMessage().contains("ETag changed"), failure.getMessage());
            assertDiscarded(dest);
        }
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"W/\"v1\"", "Wed, 21 Oct 2015 07:28:00 GMT"})
    void anUnverifiedSequentialPrefixIsNeverAppendedTo(String validator, @TempDir Path dir)
            throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            if (validator != null) server.etag("/m.gguf", validator);
            Path dest = dir.resolve("m.gguf");
            Files.writeString(dir.resolve("m.gguf.part"), "stale prefix");
            if (validator != null) Files.writeString(dir.resolve("m.gguf.part.etag"), validator);
            Fetch.download(server.url("/m.gguf"), dest, PAYLOAD.length(), null, Map.of());
            assertEquals(PAYLOAD, Files.readString(dest));
            assertNull(server.lastRange("/m.gguf"), "restart as one full response");
            assertEquals("identity", server.lastHeader("/m.gguf", "Accept-Encoding"));
        }
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "",
                "bytes 0-3/8",
                "bytes 4-6/8",
                "bytes 4-7/9",
                "bytes 4-7/*",
                "items 4-7/8",
                "bytes 4-7/999999999999999999999"
            })
    void aSequentialResumeValidatesContentRange(String range, @TempDir Path dir)
            throws IOException {
        try (FileServer server =
                FileServer.start()
                        .respond(
                                "/bad.bin",
                                exchange ->
                                        reply(
                                                exchange,
                                                range,
                                                "\"v1\"",
                                                "BBBB".getBytes(StandardCharsets.UTF_8),
                                                false))) {
            Path dest = dir.resolve("bad.bin");
            Files.writeString(dir.resolve("bad.bin.part"), "AAAA");
            Files.writeString(dir.resolve("bad.bin.part.etag"), "\"v1\"");
            assertThrows(
                    IOException.class,
                    () -> Fetch.download(server.url("/bad.bin"), dest, 8, null, Map.of()));
            assertDiscarded(dest);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void anOversizedChunkIsNotSilentlyTruncated(boolean chunked, @TempDir Path dir)
            throws IOException {
        long size = Fetch.PARALLEL_FLOOR;
        byte[] oversized = new byte[(int) (size / 2) + 1];
        try (FileServer server =
                FileServer.start()
                        .respond(
                                "/long.bin",
                                exchange -> {
                                    long start = rangeStart(exchange);
                                    reply(
                                            exchange,
                                            "bytes %d-%d/%d"
                                                    .formatted(start, start + size / 2 - 1, size),
                                            "\"v1\"",
                                            oversized,
                                            chunked);
                                })) {
            Path dest = dir.resolve("long.bin");
            assertThrows(
                    IOException.class,
                    () -> Fetch.download(server.url("/long.bin"), dest, size, null, Map.of()));
            assertDiscarded(dest);
        }
    }

    @Test
    void aValidatedParallelPrefixStillResumes(@TempDir Path dir) throws IOException {
        byte[] payload = new byte[(int) Fetch.PARALLEL_FLOOR];
        Arrays.fill(payload, (byte) 'B');
        byte[] partial = payload.clone();
        Arrays.fill(partial, partial.length / 2, partial.length, (byte) 0);
        try (FileServer server =
                FileServer.start().serve("/big.bin", payload).etag("/big.bin", "\"v1\"")) {
            Path dest = dir.resolve("big.bin");
            Files.write(dir.resolve("big.bin.part"), partial);
            Files.write(dir.resolve("big.bin.part.map"), new byte[] {1, 0});
            Files.writeString(dir.resolve("big.bin.part.etag"), "\"v1\"");
            Fetch.download(server.url("/big.bin"), dest, payload.length, null, Map.of());
            assertArrayEquals(payload, Files.readAllBytes(dest));
            assertEquals(1, server.hits("/big.bin"));
            assertEquals("\"v1\"", server.lastHeader("/big.bin", "If-Range"));
        }
    }

    @ParameterizedTest
    @CsvSource({"false, false", "false, true", "true, false", "true, true"})
    void unversionedParallelDownloadsNeedAChecksum(
            boolean checksum, boolean weakTag, @TempDir Path dir) throws IOException {
        byte[] payload = new byte[(int) Fetch.PARALLEL_FLOOR];
        new Random(19).nextBytes(payload);
        AtomicInteger fullResponses = new AtomicInteger();
        try (FileServer server =
                FileServer.start()
                        .respond(
                                "/big.bin",
                                exchange -> {
                                    String range = exchange.getRequestHeaders().getFirst("Range");
                                    String etag = weakTag ? "W/\"v1\"" : null;
                                    if (range == null) {
                                        fullResponses.incrementAndGet();
                                        reply(exchange, null, etag, payload, false);
                                    } else {
                                        int start = (int) rangeStart(exchange);
                                        int end = start + payload.length / 2;
                                        reply(
                                                exchange,
                                                "bytes %d-%d/%d"
                                                        .formatted(start, end - 1, payload.length),
                                                etag,
                                                Arrays.copyOfRange(payload, start, end),
                                                false);
                                    }
                                })) {
            Path dest = dir.resolve("big.bin");
            Fetch.download(
                    server.url("/big.bin"),
                    dest,
                    payload.length,
                    checksum ? sha256(payload) : null,
                    Map.of());
            assertArrayEquals(payload, Files.readAllBytes(dest));
            assertEquals(checksum ? 0 : 1, fullResponses.get());
        }
    }

    @Test
    void aLaterETagCannotCertifyAnUnversionedPrefix(@TempDir Path dir) throws IOException {
        String current = "B".repeat(12);
        AtomicInteger calls = new AtomicInteger();
        try (FileServer server =
                FileServer.start()
                        .respond(
                                "/m.bin",
                                exchange -> {
                                    int call = calls.incrementAndGet();
                                    if (call == 1) {
                                        // The resumed response ends early. A later refusal keeps
                                        // its partial state.
                                        reply(
                                                exchange,
                                                "bytes 4-11/12",
                                                "\"v2\"",
                                                "BBBB".getBytes(StandardCharsets.UTF_8),
                                                true);
                                    } else if (call == 2) {
                                        try (exchange) {
                                            exchange.sendResponseHeaders(403, -1);
                                        }
                                    } else {
                                        String range =
                                                exchange.getRequestHeaders().getFirst("Range");
                                        int start = range == null ? 0 : (int) rangeStart(exchange);
                                        reply(
                                                exchange,
                                                range == null ? null : "bytes " + start + "-11/12",
                                                "\"v2\"",
                                                current.substring(start)
                                                        .getBytes(StandardCharsets.UTF_8),
                                                false);
                                    }
                                })) {
            Path dest = dir.resolve("m.bin");
            Files.writeString(dir.resolve("m.bin.part"), "AAAA");
            assertThrows(
                    IOException.class,
                    () ->
                            Fetch.download(
                                    server.url("/m.bin"), dest, 12, sha256(current), Map.of()));
            assertTrue(
                    Files.notExists(dir.resolve("m.bin.part.etag")),
                    "old prefix has no proven ETag");
            Fetch.download(server.url("/m.bin"), dest, 12, null, Map.of());
            assertEquals(current, Files.readString(dest));
        }
    }

    @Test
    void unexpectedContentEncodingIsRefused(@TempDir Path dir) throws IOException {
        try (FileServer server =
                FileServer.start()
                        .respond(
                                "/encoded.bin",
                                exchange -> {
                                    exchange.getResponseHeaders().set("Content-Encoding", "gzip");
                                    reply(exchange, null, "\"v1\"", new byte[] {1, 2, 3}, false);
                                })) {
            Path dest = dir.resolve("encoded.bin");
            var failure =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/encoded.bin"), dest, 3, null, Map.of()));
            assertTrue(failure.getMessage().contains("Content-Encoding"));
            assertDiscarded(dest);
        }
    }

    @ParameterizedTest
    @ValueSource(longs = {-1, 8})
    void aValidatedSequentialPrefixResumesWithKnownOrUnknownSize(long size, @TempDir Path dir)
            throws IOException {
        AtomicReference<String> request = new AtomicReference<>();
        try (FileServer server =
                FileServer.start()
                        .respond(
                                "/m.bin",
                                exchange -> {
                                    request.set(
                                            exchange.getRequestHeaders().getFirst("Range")
                                                    + " "
                                                    + exchange.getRequestHeaders()
                                                            .getFirst("If-Range"));
                                    reply(
                                            exchange,
                                            "Bytes 4-7/8",
                                            "\"v1\"",
                                            "BBBB".getBytes(StandardCharsets.UTF_8),
                                            false);
                                })) {
            Path dest = dir.resolve("m.bin");
            Files.writeString(dir.resolve("m.bin.part"), "AAAA");
            Files.writeString(dir.resolve("m.bin.part.etag"), "\"v1\"");
            Fetch.download(server.url("/m.bin"), dest, size, null, Map.of());
            assertEquals("AAAABBBB", Files.readString(dest));
            assertEquals("bytes=4- \"v1\"", request.get());
        }
    }

    private static long rangeStart(HttpExchange exchange) {
        return Long.parseLong(
                exchange.getRequestHeaders()
                        .getFirst("Range")
                        .substring("bytes=".length())
                        .split("-", 2)[0]);
    }

    private static void reply(
            HttpExchange exchange, String range, String etag, byte[] bytes, boolean chunked)
            throws IOException {
        try (exchange) {
            if (range != null && !range.isEmpty())
                exchange.getResponseHeaders().set("Content-Range", range);
            if (etag != null) exchange.getResponseHeaders().set("ETag", etag);
            exchange.sendResponseHeaders(range == null ? 200 : 206, chunked ? 0 : bytes.length);
            exchange.getResponseBody().write(bytes);
        }
    }

    private static void assertDiscarded(Path dest) {
        assertTrue(Files.notExists(dest), "corrupt data must not be published");
        for (String suffix : new String[] {".part", ".part.map", ".part.etag"}) {
            assertTrue(Files.notExists(dest.resolveSibling(dest.getFileName() + suffix)), suffix);
        }
    }
}
