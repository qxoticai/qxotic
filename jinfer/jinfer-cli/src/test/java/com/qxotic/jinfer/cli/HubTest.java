package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.hub.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HubTest {
    @TempDir Path dir;

    @Test
    void pullCachesInArgumentOrderAndForceRefreshes() throws Exception {
        AtomicInteger fetches = new AtomicInteger();
        var source =
                new ModelSource() {
                    public boolean supports(ModelRef ref) {
                        return true;
                    }

                    public List<RemoteFile> list(ModelRef ref, String directory) {
                        return List.of(new RemoteFile("model-Q8_0.gguf", 4, null));
                    }

                    public void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException {
                        fetches.incrementAndGet();
                        Files.createDirectories(into.getParent());
                        Files.write(into, new byte[] {'G', 'G', 'U', 'F'});
                    }
                };
        ModelStore store = ModelStore.of(dir, source);
        var capture = new CliFixtures.Capture("");
        String a = "owner/first:Q8_0", b = "owner/second:Q8_0";
        assertEquals(0, Main.run(new String[] {"pull", a, b}, capture.io, store), capture.err());
        var paths = capture.out().lines().toList();
        assertEquals(2, paths.size());
        assertTrue(paths.get(0).contains("first"));
        assertTrue(paths.get(1).contains("second"));
        assertEquals(2, fetches.get());
        Hub.pull(List.of(a, b), false, store, capture.io.out());
        assertEquals(2, fetches.get());
        Hub.pull(List.of(a), true, store, capture.io.out());
        assertEquals(3, fetches.get());

        var listing = new CliFixtures.Capture("");
        assertEquals(0, Main.run(new String[] {"list"}, listing.io, store));
        assertTrue(listing.out().contains("first"));
        assertTrue(listing.out().contains("second"));
        assertEquals(3, fetches.get(), "listing never fetches");
    }

    @Test
    void emptyListAndBadHubArgumentsArePredictable() {
        var capture = new CliFixtures.Capture("");
        var store = ModelStore.of(dir);
        Hub.list(List.of(), dir, capture.io.out());
        assertTrue(capture.out().contains("no models cached"));
        for (String[] args :
                new String[][] {{"pull"}, {"pull", "--wat"}, {"list", "extra"}, {"cache-info"}})
            assertEquals(2, Main.run(args, capture.io, store));
    }

    @Test
    void cacheInfoUsesTheActualPromptCacheFormat() throws Exception {
        Path file = dir.resolve("prompts.jkv");
        FrozenBlocks.createEmpty(file, new ContentKey("test"));
        var capture = new CliFixtures.Capture("");
        assertEquals(
                0,
                Main.run(
                        new String[] {"cache-info", file.toString()},
                        capture.io,
                        ModelStore.of(dir)),
                capture.err());
        assertEquals(FrozenBlocks.describe(file), capture.out());
        Files.writeString(file, "corrupt");
        assertEquals(
                0,
                Main.run(
                        new String[] {"cache-info", file.toString()},
                        capture.io,
                        ModelStore.of(dir)));
        assertTrue(capture.out().contains("not a frozen prompt cache"));
    }

    @Test
    void failedDownloadsReturnAnErrorWithoutPrintingAResultPath() {
        var source =
                new ModelSource() {
                    public boolean supports(ModelRef ref) {
                        return true;
                    }

                    public List<RemoteFile> list(ModelRef ref, String directory)
                            throws IOException {
                        throw new IOException("offline test failure");
                    }

                    public void fetch(ModelRef ref, RemoteFile file, Path into) {
                        throw new AssertionError("listing failed");
                    }
                };
        var capture = new CliFixtures.Capture("");
        assertEquals(
                1,
                Main.run(
                        new String[] {"pull", "owner/repo:Q8_0"},
                        capture.io,
                        ModelStore.of(dir, source)));
        assertEquals("", capture.out());
        assertTrue(capture.err().contains("offline test failure"), capture.err());
    }

    @Test
    void humanSizesAndReferencesRemainUsableInAListing() {
        var capture = new CliFixtures.Capture("");
        Hub.list(
                List.of(
                        new ModelStore.Cached("owner/small/model.gguf", 512),
                        new ModelStore.Cached("owner/big/model.gguf", 2L << 30)),
                dir,
                capture.io.out());
        assertTrue(capture.out().contains("512 B"));
        assertTrue(capture.out().contains("2.0 GB"), capture.out());
        assertTrue(capture.out().contains("owner/big/model.gguf"));
    }
}
