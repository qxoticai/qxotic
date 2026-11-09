package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.hub.ModelRef;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.TokenizerLoadException;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The source grammar, row by row: what each spelling resolves to, and how it fails. */
class SourcesTest {
    @TempDir Path root;
    @TempDir Path disk;

    /** A GGUF header as the writer lays it out, followed by weights nobody should download. */
    static byte[] gguf() throws IOException {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.architecture", "test")
                        .putArrayOfString("tokenizer.ggml.tokens", new String[] {"<s>", "hello"})
                        .putTensor(TensorEntry.create("weight", new long[] {4}, GGMLType.F32, 0))
                        .build();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        GGUF.write(gguf, Channels.newChannel(out));
        byte[] header = out.toByteArray();
        byte[] weights = "weights ".repeat(1024).getBytes(StandardCharsets.UTF_8);
        byte[] model = Arrays.copyOf(header, header.length + weights.length);
        System.arraycopy(weights, 0, model, header.length, weights.length);
        return model;
    }

    private ModelStore store(FakeSource fake) {
        return ModelStore.of(root, fake);
    }

    // ---- paths ----

    @Test
    void aFileOnDiskIsAFileAndAFolderAFolder() throws IOException {
        Path file = Files.writeString(disk.resolve("sha256-3f1a"), "anything");
        assertEquals(
                Sources.Located.file(file),
                Sources.locate(file.toString(), store(new FakeSource())));
        assertEquals(
                Sources.Located.folder(disk),
                Sources.locate(disk.toString(), store(new FakeSource())));
    }

    @Test
    void aByteOrderMarkIsRefusedByName() throws Exception {
        // as Hugging Face and Python's json refuse it, not "is not a tokenizer"
        Path bom = disk.resolve("bom.json");
        Files.write(bom, "\uFEFF{\"model\":{}}".getBytes(StandardCharsets.UTF_8));
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.load(bom.toString(), store(new FakeSource())));
        assertEquals(Errors.EINVAL, e.code);
        assertTrue(
                e.getMessage()
                        .endsWith(
                                "starts with a UTF-8 byte order mark, which JSON does not allow:"
                                        + " save it without one"),
                e.getMessage());
    }

    @Test
    void aFifoIsRefusedNotOpened() throws Exception {
        assumeFalse(System.getProperty("os.name").startsWith("Windows"), "no FIFOs on Windows");
        Path fifo = disk.resolve("fifo");
        assertEquals(0, new ProcessBuilder("mkfifo", fifo.toString()).start().waitFor());
        // on a thread of its own: the regression blocks in open(), which no interrupt ends
        Errors.Failure e =
                assertTimeoutPreemptively(
                        Duration.ofSeconds(10),
                        () ->
                                assertThrows(
                                        Errors.Failure.class,
                                        () ->
                                                Sources.load(
                                                        fifo.toString(), store(new FakeSource()))));
        assertEquals(Errors.EINVAL, e.code);
        assertTrue(e.getMessage().endsWith("is not a regular file"), e.getMessage());
    }

    @Test
    void aMissingPathIsNotFound() {
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () ->
                                Sources.locate(
                                        disk.resolve("none").toString(), store(new FakeSource())));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertTrue(e.getMessage().contains("does not exist"));
        assertFalse(e.getMessage().contains("~"), "a hint only where a ~ was written");
    }

    @Test
    void aTildeIsAFolderNameAndAMissingOneSaysItIsNotExpanded() {
        // no shell reads a C string: ~/x is the path as given, never a hub reference nor $HOME/x
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () ->
                                Sources.locate(
                                        "~/ctest-none/tokenizer.json", store(new FakeSource())));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertEquals(
                "~/ctest-none/tokenizer.json does not exist (~ is not expanded: write the full"
                        + " path)",
                e.getMessage());
    }

    @Test
    void contentDecidesTheFormat() throws IOException {
        assertEquals(Sources.Kind.GGUF, Sources.sniff(Files.write(disk.resolve("blob"), gguf())));
        assertEquals(
                Sources.Kind.JSON,
                Sources.sniff(Files.writeString(disk.resolve("tok"), "  \n{\"model\": {}}")));
        assertEquals(
                Sources.Kind.UNKNOWN,
                Sources.sniff(Files.writeString(disk.resolve("text"), "hello")));
    }

    // ---- references ----

    @Test
    void aQuantNamesTheGgufForTheHeaderReader() {
        FakeSource fake = new FakeSource().serving("", "thing-Q8_0.gguf", "thing-Q4_K_M.gguf");
        Sources.Located at = Sources.locate("hf.co/t/thing:Q8_0", store(fake));
        assertEquals("thing-Q8_0.gguf", at.ggufFile());
        assertEquals("thing", at.repository().repo());
        assertFalse(fake.fetched, "the store names the file; the reader fetches the header");
    }

    @Test
    void offlineAQuantReachesTheHeaderItSelectedOnline() {
        Sources.locate(
                "hf.co/t/thing:Q8_0", store(new FakeSource().serving("", "thing-Q8_0.gguf")));
        Sources.Located at = Sources.locate("t/thing:Q8_0", ModelStore.of(root)); // no source
        assertEquals("thing-Q8_0.gguf", at.ggufFile());
    }

    @Test
    void aGgufPathIsHandedOverAsIs() {
        FakeSource fake = new FakeSource().serving("sub", "sub/other.gguf");
        Sources.Located at = Sources.locate("t/thing@v2/sub/other.gguf", store(fake));
        assertEquals("sub/other.gguf", at.ggufFile());
        assertEquals("v2", at.repository().revision());
        assertFalse(fake.fetched);
    }

    @Test
    void aGgufJinferHasIsReadInPlace() throws IOException {
        Path whole = root.resolve("hf.co/t/thing/thing-Q8_0.gguf");
        Files.createDirectories(whole.getParent());
        Files.write(whole, gguf());
        assertEquals(
                Sources.Located.file(whole), Sources.locate("t/thing:Q8_0", ModelStore.of(root)));
    }

    @Test
    void aRepositoryRootIsItsTokenizerJson() throws IOException {
        FakeSource fake = new FakeSource().serving("", "tokenizer.json", "thing-Q8_0.gguf");
        Sources.Located at = Sources.locate("t/thing", store(fake));
        assertEquals("tokenizer.json", at.path().getFileName().toString());
        assertEquals("{}", Files.readString(at.path()));
        assertFalse(at.folder());
    }

    @Test
    void aFolderPathIsItsTokenizerJson() throws IOException {
        FakeSource fake = new FakeSource().serving("sub", "sub/tokenizer.json");
        Sources.Located at = Sources.locate("hf.co/t/thing/sub", store(fake));
        assertEquals(root.resolve("hf.co/t/thing/sub/tokenizer.json"), at.path());
    }

    @Test
    void aRootWithoutTokenizerJsonNamesWhatItHolds() {
        FakeSource fake = new FakeSource().serving("", "thing-Q8_0.gguf", "thing-Q4_K_M.gguf");
        Errors.Failure e =
                assertThrows(Errors.Failure.class, () -> Sources.locate("t/thing", store(fake)));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertTrue(e.getMessage().contains("thing-Q8_0.gguf"), e.getMessage());
        assertFalse(fake.fetched, "a bare reference never opens a GGUF");
    }

    @Test
    void aTiktokenRepositoryGoesToTheLoadersOwnHubPath() {
        FakeSource fake = new FakeSource().serving("", "tiktoken.model", "tokenizer_config.json");
        Sources.Located at = Sources.locate("t/thing@v1", store(fake));
        assertNull(at.path());
        assertNull(at.ggufFile());
        assertEquals("t", at.repository().owner());
        assertEquals("v1", at.repository().revision());
    }

    @Test
    void anotherHubFileIsFetchedWhole() {
        FakeSource fake = new FakeSource().serving("", "vocab.json");
        Sources.Located at = Sources.locate("t/thing/vocab.json", store(fake));
        assertEquals("vocab.json", at.path().getFileName().toString());
        assertTrue(fake.fetched);
    }

    @Test
    void aMissingFileInARepositoryIsNotFound() {
        FakeSource fake = new FakeSource().serving("", "tokenizer.json");
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.locate("t/thing/other.json", store(fake)));
        assertEquals(Errors.ENOTFOUND, e.code);
    }

    @Test
    void aMissingBareReferenceWithAPathSaysHowAPathIsWritten() {
        FakeSource fake = new FakeSource().serving("", "tokenizer.json");
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.load("t/thing/other.json", store(fake)));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertTrue(
                e.getMessage().endsWith("; a local path is written './t/thing/other.json'"),
                e.getMessage());
        // a host-qualified reference spells no path: the hub's answer as it is
        Errors.Failure qualified =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.load("hf.co/t/thing/other.json", store(fake)));
        assertEquals(Errors.ENOTFOUND, qualified.code);
        assertFalse(qualified.getMessage().contains("a local path"), qualified.getMessage());
    }

    @Test
    void aLoaderFetchFailsInTheHeadersCodes() {
        ModelRef ref = ModelRef.parse("t/thing");
        IOException gated =
                new IOException("[huggingface] Failed to download https://x/y (HTTP 401)");
        Errors.Failure e =
                assertThrows(Errors.Failure.class, () -> fetchFailing(ref, false, gated));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertTrue(
                e.getMessage().startsWith("t/thing does not exist, or is gated or private"),
                e.getMessage());
        assertTrue(e.getMessage().endsWith("set HF_TOKEN"), e.getMessage());
        FileNotFoundException missing =
                new FileNotFoundException("[huggingface] HTTP 404 (cached): https://x/y");
        assertEquals(
                Errors.ENOTFOUND,
                assertThrows(Errors.Failure.class, () -> fetchFailing(ref, false, missing)).code);
        AccessDeniedException denied =
                new AccessDeniedException(root.resolve("repository-artifacts").toString());
        e = assertThrows(Errors.Failure.class, () -> fetchFailing(ref, false, denied));
        assertEquals(Errors.ERROR, e.code);
        assertEquals("cache root is not writable: " + denied.getFile(), e.getMessage());
        // offline, a cache folder the loader cannot make holds nothing: not cached, as a miss is
        e = assertThrows(Errors.Failure.class, () -> fetchFailing(ref, true, denied));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertTrue(e.getMessage().contains("not cached"), e.getMessage());
        // anything else is the loader's own failure, as it says it
        IOException down =
                new IOException("[huggingface] Failed to download https://x/y (HTTP 503)");
        assertSame(
                down,
                assertThrows(TokenizerLoadException.class, () -> fetchFailing(ref, false, down))
                        .getCause());
    }

    private static Tokenizer fetchFailing(ModelRef ref, boolean offline, IOException cause) {
        throw Sources.failure("t/thing", ref, offline, new TokenizerLoadException("failed", cause));
    }

    @Test
    void aFolderWithoutATokenizerJsonIsNotFound() {
        Errors.Failure e = assertThrows(Errors.Failure.class, () -> Sources.load(disk.toString()));
        assertEquals(Errors.ENOTFOUND, e.code);
        // "is a folder" says what the path is, even when it is named tokenizer.json
        assertTrue(
                e.getMessage().contains("is a folder that holds no tokenizer.json"),
                e.getMessage());
    }

    @Test
    void aRefAUrlWouldMisreadIsInvalidBeforeAnyFetch() {
        for (String bad : new String[] {"a b/c", "t/thing#x", "t/thing/x-Q8_0.gguf:Q8_0"}) {
            FakeSource fake = new FakeSource();
            Errors.Failure e =
                    assertThrows(Errors.Failure.class, () -> Sources.locate(bad, store(fake)), bad);
            assertEquals(Errors.EINVAL, e.code, bad + ": " + e.getMessage());
            assertTrue(fake.listed == 0 && !fake.fetched, bad + ": the hub was asked");
        }
    }

    @Test
    void aPathAndAQuantTogetherAreInvalid() {
        FakeSource fake = new FakeSource().serving("", "thing-Q8_0.gguf");
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.locate("t/thing/thing-Q8_0.gguf:Q8_0", store(fake)));
        assertEquals(Errors.EINVAL, e.code);
    }

    @Test
    void aMisspelledHostIsInvalidNotAnOwner() {
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.locate("huggingface.com/t/thing", store(new FakeSource())));
        assertEquals(Errors.EINVAL, e.code);
        assertTrue(e.getMessage().contains("modelscope.cn"), e.getMessage());
    }

    @Test
    void offlineServesTheCacheOnly() {
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.locate("t/thing:Q8_0", ModelStore.of(root)));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertTrue(e.getMessage().contains("offline"), e.getMessage());
    }

    @Test
    void offlineTouchesNothingOnAnyHubPath() {
        FakeSource fake =
                new FakeSource()
                        .serving("", "thing-Q8_0.gguf", "tokenizer.json", "vocab.json")
                        .serving("sub", "sub/tokenizer.json");
        System.setProperty(
                "jinfer.offline", "true"); // the store's own knob: a store WITH sources, locked
        try {
            for (String source :
                    new String[] {"t/thing:Q8_0", "t/thing", "t/thing/sub", "t/thing/vocab.json"}) {
                // a plain ref goes on to the tiktoken loader's own cache, which is empty too
                Errors.Failure e =
                        assertThrows(
                                Errors.Failure.class,
                                () -> Sources.load(source, store(fake)),
                                source);
                assertEquals(Errors.ENOTFOUND, e.code, source);
                assertTrue(e.getMessage().contains("offline"), e.getMessage());
            }
            // a .gguf path needs no lookup: it is handed to the header reader, whose offline
            // transport refuses, which is not cached, and a path spelled bare is a hint
            assertEquals(
                    "thing-Q8_0.gguf",
                    Sources.locate("t/thing/thing-Q8_0.gguf", store(fake)).ggufFile());
            System.setProperty("toknroll.cache.root", root.toString()); // the reader's own cache
            Errors.Failure e =
                    assertThrows(
                            Errors.Failure.class,
                            () -> Sources.load("t/thing/thing-Q8_0.gguf", store(fake)));
            assertEquals(Errors.ENOTFOUND, e.code);
            assertEquals(
                    "t/thing/thing-Q8_0.gguf is not cached, and offline mode never fetches; a local"
                            + " path is written './t/thing/thing-Q8_0.gguf'",
                    e.getMessage());
        } finally {
            System.clearProperty("jinfer.offline");
            System.clearProperty("toknroll.cache.root");
        }
        assertEquals(0, fake.listed, "no listing");
        assertFalse(fake.fetched, "no download");
    }

    @Test
    void anEmptySourceIsInvalid() {
        assertEquals(
                Errors.EINVAL,
                assertThrows(
                                Errors.Failure.class,
                                () -> Sources.locate("  ", store(new FakeSource())))
                        .code);
    }
}
