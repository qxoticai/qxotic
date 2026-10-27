package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.hub.ModelStore;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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
    void aMissingPathIsNotFound() {
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () ->
                                Sources.locate(
                                        disk.resolve("none").toString(), store(new FakeSource())));
        assertEquals(Errors.ENOTFOUND, e.code);
        assertTrue(e.getMessage().contains("does not exist"));
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
    void aPathAndAQuantTogetherAreInvalid() {
        FakeSource fake = new FakeSource().serving("", "thing-Q8_0.gguf");
        Errors.Failure e =
                assertThrows(
                        Errors.Failure.class,
                        () -> Sources.locate("t/thing/thing-Q8_0.gguf:Q8_0", store(fake)));
        assertEquals(Errors.EINVAL, e.code);
    }

    @Test
    void theOldPrefixesSayWhatToWriteInstead() {
        for (String old : new String[] {"hf:Qwen/Qwen3-0.6B", "ms:Qwen/Qwen3-0.6B"}) {
            Errors.Failure e =
                    assertThrows(
                            Errors.Failure.class,
                            () -> Sources.locate(old, store(new FakeSource())));
            assertEquals(Errors.EINVAL, e.code);
            assertTrue(e.getMessage().contains("hf.co/owner/repo"), e.getMessage());
        }
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
                Errors.Failure e =
                        assertThrows(
                                Errors.Failure.class,
                                () -> Sources.locate(source, store(fake)),
                                source);
                assertEquals(Errors.ENOTFOUND, e.code, source);
            }
            // a .gguf path needs no lookup: it is handed to the header reader, whose offline
            // transport refuses
            assertEquals(
                    "thing-Q8_0.gguf",
                    Sources.locate("t/thing/thing-Q8_0.gguf", store(fake)).ggufFile());
        } finally {
            System.clearProperty("jinfer.offline");
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
