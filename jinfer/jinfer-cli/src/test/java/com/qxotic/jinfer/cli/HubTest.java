package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.hub.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HubTest {
    @TempDir Path dir;

    @Test
    void pullCachesInArgumentOrderAndForceRefreshes(@TempDir Path elsewhere) throws Exception {
        AtomicInteger fetches = new AtomicInteger();
        String sha256 =
                HexFormat.of()
                        .formatHex(
                                MessageDigest.getInstance("SHA-256")
                                        .digest(new byte[] {'G', 'G', 'U', 'F'}));
        var source =
                new ModelSource() {
                    public boolean supports(ModelRef ref) {
                        return true;
                    }

                    public List<RemoteFile> list(ModelRef ref, String directory) {
                        return List.of(new RemoteFile("model-Q8_0.gguf", 4, sha256));
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
        // each name is the shortest ref that loads the file, ready to paste back
        assertTrue(listing.out().replace("\r\n", "\n").contains("  " + a + "\n"), listing.out());
        assertTrue(listing.out().replace("\r\n", "\n").contains("  " + b + "\n"), listing.out());
        assertEquals(3, fetches.get(), "listing never fetches");

        var found = new CliFixtures.Capture("");
        assertEquals(0, Main.run(new String[] {"find", b, a}, found.io, store), found.err());
        assertEquals(List.of(paths.get(1), paths.get(0)), found.out().lines().toList());
        assertEquals(3, fetches.get(), "find never fetches");

        var removed = new CliFixtures.Capture("");
        assertEquals(0, Main.run(new String[] {"remove", a}, removed.io, store), removed.err());
        assertEquals("removed model-Q8_0.gguf (4 B)\n", removed.out().replace("\r\n", "\n"));
        assertFalse(Files.exists(Path.of(paths.get(0))));
        assertTrue(Files.exists(Path.of(paths.get(1))), "remove deletes only what it names");
        var missing = new CliFixtures.Capture("");
        assertEquals(1, Main.run(new String[] {"find", a}, missing.io, store));
        assertEquals("", missing.out());
        assertTrue(missing.err().contains("not cached: " + a), missing.err());
        // one missing or outside argument fails the command, yet the others still go
        Path local = Files.writeString(elsewhere.resolve("local.gguf"), "mine");
        var mixed = new CliFixtures.Capture("");
        assertEquals(1, Main.run(new String[] {"remove", a, local.toString(), b}, mixed.io, store));
        assertEquals("removed model-Q8_0.gguf (4 B)\n", mixed.out().replace("\r\n", "\n"));
        assertTrue(mixed.err().contains("not cached: " + a), mixed.err());
        assertTrue(mixed.err().contains("is not in the model cache"), mixed.err());
        assertTrue(Files.exists(local), "a file outside the cache is never deleted");
        assertFalse(Files.exists(Path.of(paths.get(1))));
    }

    /** find answers what -m would load: a browser URL too, and a bare word gets the ref shape. */
    @Test
    void findTakesWhatTheModelOptionTakes() throws Exception {
        Path file = dir.resolve("hf.co/acme/m/m-Q8_0.gguf");
        Files.createDirectories(file.getParent());
        Files.writeString(file, "GGUF");
        var store = ModelStore.of(dir);
        var found = new CliFixtures.Capture("");
        String url = "https://huggingface.co/acme/m/blob/main/m-Q8_0.gguf";
        assertEquals(0, Main.run(new String[] {"find", url}, found.io, store), found.err());
        assertEquals(file.toString(), found.out().strip());

        var word = new CliFixtures.Capture("");
        assertEquals(1, Main.run(new String[] {"find", "acme", "acme:"}, word.io, store));
        assertTrue(
                word.err()
                        .contains("no such file: 'acme'; a model reference is owner/repo[:quant]"),
                word.err());
        assertTrue(word.err().contains("acme:"), word.err());
    }

    /** A name keeps what follows the model's own name, not just the shortest quant that loads. */
    @Test
    void aNameKeepsTheWholeQuant() throws Exception {
        Path file = dir.resolve("hf.co/acme/Solo-GGUF/Solo-UD-Q4_K_M.gguf");
        Files.createDirectories(file.getParent());
        Files.writeString(file, "GGUF");
        assertEquals(
                "acme/Solo-GGUF:UD-Q4_K_M",
                Hub.name(ModelStore.of(dir), "hf.co/acme/Solo-GGUF/Solo-UD-Q4_K_M.gguf"));
    }

    /** A repository holding several cached files is never a guess: both commands list them. */
    @Test
    void aRepositoryWithSeveralCachedFilesAsksForOne() throws Exception {
        for (String name : List.of("m-Q4_0.gguf", "m-QAD-Q4_0.gguf", "m-Q8_0.gguf")) {
            Path file = dir.resolve("hf.co/acme/m/" + name);
            Files.createDirectories(file.getParent());
            Files.writeString(file, "GGUF");
        }
        var store = ModelStore.of(dir);
        assertEquals("acme/m:Q8_0", Hub.name(store, "hf.co/acme/m/m-Q8_0.gguf"));
        assertEquals(
                "acme/m:QAD-Q4_0",
                Hub.name(store, "hf.co/acme/m/m-QAD-Q4_0.gguf"),
                "Q4_0 alone would pick two files");
        assertEquals(
                "acme/m/m-Q4_0.gguf",
                Hub.name(store, "hf.co/acme/m/m-Q4_0.gguf"),
                "no quant names it alone: the path does");

        var miss = new CliFixtures.Capture("");
        assertEquals(1, Main.run(new String[] {"find", "acme/m"}, miss.io, store));
        assertTrue(
                miss.err().contains("acme/m/m-Q4_0.gguf\n  acme/m:Q8_0\n  acme/m:QAD-Q4_0"),
                miss.err());

        var refused = new CliFixtures.Capture("");
        assertEquals(1, Main.run(new String[] {"remove", "acme/m"}, refused.io, store));
        assertEquals("", refused.out());
        assertTrue(
                refused.err().contains("acme/m holds 3 cached files; name the one to remove:"),
                refused.err());
        assertEquals(3, store.cached().stream().filter(c -> c.ref().contains("acme/m")).count());

        var removed = new CliFixtures.Capture("");
        String[] each = {"remove", "acme/m/m-Q4_0.gguf", "acme/m:Q8_0", "acme/m:QAD-Q4_0"};
        assertEquals(0, Main.run(each, removed.io, store), removed.err());
        assertEquals(3, removed.out().lines().count(), removed.out());
        assertTrue(Files.notExists(dir.resolve("hf.co")), "nothing left behind");
    }

    @Test
    void failedForcePullPreservesTheFlatCache() throws Exception {
        Path cached = dir.resolve("hf.co/owner/repo/model-Q8_0.gguf");
        Files.createDirectories(cached.getParent());
        Files.writeString(cached, "working model");
        var source =
                new ModelSource() {
                    public boolean supports(ModelRef ref) {
                        return true;
                    }

                    public List<RemoteFile> list(ModelRef ref, String directory)
                            throws IOException {
                        throw new IOException("repository unavailable");
                    }

                    public void fetch(ModelRef ref, RemoteFile file, Path into) {
                        throw new AssertionError("listing failed");
                    }
                };
        ModelStore store = ModelStore.of(dir, source);
        var capture = new CliFixtures.Capture("");
        assertEquals(
                1,
                Main.run(new String[] {"pull", "--force", "owner/repo:Q8_0"}, capture.io, store));
        assertEquals("", capture.out());
        assertTrue(capture.err().contains("repository unavailable"), capture.err());
        assertTrue(Files.exists(cached), "failed refresh deleted the working model");
        assertEquals("working model", Files.readString(cached));
        assertEquals(cached, store.resolve("owner/repo:Q8_0"));
    }

    @Test
    void emptyListAndBadHubArgumentsArePredictable() {
        var capture = new CliFixtures.Capture("");
        var store = ModelStore.of(dir);
        Hub.list(List.of(), ref -> ref, dir, capture.io.out());
        assertTrue(capture.out().contains("no models cached"));
        for (String[] args :
                new String[][] {
                    {"pull"}, {"pull", "--wat"}, {"list", "extra"}, {"find"}, {"cache-info"}
                }) assertEquals(2, Main.run(args, capture.io, store));
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
        // a file that never was a cache is refused with the exit code a script can act on
        Files.writeString(file, "corrupt");
        var junk = new CliFixtures.Capture("");
        assertEquals(
                1,
                Main.run(
                        new String[] {"cache-info", file.toString()}, junk.io, ModelStore.of(dir)));
        assertEquals("", junk.out());
        assertTrue(junk.err().contains("not a valid frozen prompt cache"), junk.err());
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

    /** Sizes lead in a fixed column; a name of any width follows, so nothing can misalign. */
    @Test
    void humanSizesAndReferencesRemainUsableInAListing() {
        var capture = new CliFixtures.Capture("");
        Hub.list(
                List.of(
                        new ModelStore.Cached("owner/small/model.gguf", 512),
                        new ModelStore.Cached("owner/big/model.gguf", 2L << 30),
                        new ModelStore.Cached("hf.co/団体/モデル/量子化-Q4_0.gguf", 3L << 20),
                        new ModelStore.Cached("/models/Édouard's café.gguf", 1)),
                ref -> ref,
                dir,
                capture.io.out());
        List<String> lines = capture.out().replace("\r\n", "\n").lines().toList();
        assertEquals(5, lines.size(), capture.out());
        for (String line : lines) {
            assertTrue(line.substring(0, 10).matches(" *[0-9.]+ [KMGT]?B"), line);
            assertEquals("  ", line.substring(10, 12), line);
        }
        // alphabetical, ignoring case
        assertTrue(lines.get(0).endsWith("  /models/Édouard's café.gguf"), lines.get(0));
        assertTrue(lines.get(1).endsWith("  hf.co/団体/モデル/量子化-Q4_0.gguf"), lines.get(1));
        assertTrue(lines.get(3).endsWith("  owner/small/model.gguf"), lines.get(3));
        assertTrue(capture.out().contains("2.0 GB"), capture.out());
        assertTrue(lines.getLast().endsWith("  total"), lines.getLast());
    }
}
