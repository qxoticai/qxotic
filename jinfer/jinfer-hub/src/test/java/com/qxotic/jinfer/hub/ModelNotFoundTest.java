package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Every way a source names nothing is one type, so a caller tells "not there" from "my code passed
 * a bad value"; a source that names something wrongly stays a plain IllegalArgumentException.
 */
class ModelNotFoundTest {
    private static final RemoteFile Q8 = new RemoteFile("thing-Q8_0.gguf", 7, null);

    @Test
    void aMissingLocalFile(@TempDir Path root) {
        String path = root.resolve("absent.gguf").toString();
        var e = assertThrows(ModelNotFoundException.class, () -> ModelStore.of(root).resolve(path));
        assertEquals(path, e.source());
    }

    @Test
    void aRepositoryTheHubRefuses() throws IOException {
        for (int status : new int[] {401, 403, 404}) {
            HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext(
                    "/",
                    exchange -> {
                        exchange.sendResponseHeaders(status, -1);
                        exchange.close();
                    });
            server.start();
            try {
                URI hub = URI.create("http://127.0.0.1:" + server.getAddress().getPort());
                var hf = new RepositorySource(ModelRef.Host.HF, hub);
                var e =
                        assertThrows(
                                ModelNotFoundException.class,
                                () -> hf.commitFor(ModelRef.parse("owner/repo")));
                assertEquals(ModelRef.parse("owner/repo").toString(), e.source());
                assertInstanceOf(Fetch.HttpStatusException.class, e.getCause(), "the hub's answer");
            } finally {
                server.stop(0);
            }
        }
    }

    @Test
    void aQuantOrFileTheRepositoryLacks(@TempDir Path root) {
        ModelStore store = ModelStore.of(root, new FakeSource("fake").serving("", Q8));
        var quant =
                assertThrows(
                        ModelNotFoundException.class,
                        () -> store.resolve("hf.co/acme/thing:Q4_K_M"));
        assertEquals("hf.co/acme/thing:Q4_K_M", quant.source());
        assertThrows(
                ModelNotFoundException.class, () -> store.resolve("hf.co/acme/thing/absent.gguf"));
    }

    /** Any source's "no" (ModelSource's contract) leaves the store as the one type, cause kept. */
    @Test
    void aSourcesNoIsNotFound(@TempDir Path root) {
        IOException answer = new IOException("HTTP 404");
        ModelStore store =
                ModelStore.of(
                        root,
                        new FakeSource("custom")
                                .failing(new IllegalArgumentException("no acme/thing", answer)));
        var e =
                assertThrows(
                        ModelNotFoundException.class, () -> store.resolve("hf.co/acme/thing:Q8_0"));
        assertEquals(answer, e.getCause());
        var own = new FakeSource("custom").failing(new IllegalArgumentException("not a quant: Q9"));
        var refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, own).resolve("hf.co/acme/thing:Q8_0"));
        assertFalse(
                refused instanceof ModelNotFoundException, "the ref's own fault stays as it is");
    }

    @Test
    void anOfflineMiss(@TempDir Path root) {
        System.setProperty("jinfer.offline", "true");
        try {
            ModelStore store = ModelStore.of(root, new FakeSource("fake").serving("", Q8));
            var e =
                    assertThrows(
                            ModelNotFoundException.class,
                            () -> store.resolve("hf.co/acme/thing:Q8_0"));
            assertEquals("hf.co/acme/thing:Q8_0", e.source());
        } finally {
            System.clearProperty("jinfer.offline");
        }
    }

    @Test
    void anAmbiguousQuantIsTheArgumentsFault(@TempDir Path root) {
        RemoteFile other = new RemoteFile("other-Q8_0.gguf", 7, null);
        ModelStore store = ModelStore.of(root, new FakeSource("fake").serving("", Q8, other));
        var e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> store.resolve("hf.co/acme/thing:Q8_0"));
        assertFalse(e instanceof ModelNotFoundException, e.getMessage());
    }
}
