package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link Models} dispatch through the {@link RecordingProvider} on the test classpath: what a port
 * receives for each kind, what is refused before a port is called, and how an embedded GGUF
 * arrives.
 */
class ModelsDispatchTest {

    @TempDir Path dir;

    @BeforeEach
    void reset() {
        RecordingProvider.reset();
    }

    private static Builder fake() {
        return Builder.newBuilder().putString("general.architecture", "fake");
    }

    private Path write(Builder builder, String name) throws IOException {
        Path file = dir.resolve(name);
        GGUF.write(builder.build(), file);
        return file;
    }

    @Test
    void theTestProviderIsDiscovered() {
        assertTrue(
                Models.supportedArchitectures().containsAll(java.util.Set.of("fake", "fake-too")));
    }

    @Test
    void languageLoadHandsThePortTheFileAndItsAttachments() throws IOException {
        Path model = write(fake().putString("tokenizer.chat_template", "T"), "m.gguf");
        Path media = Files.write(dir.resolve("mmproj.gguf"), new byte[] {1, 2, 3});
        Tokenizer tokenizer = RecordingProvider.tokenizer(0);

        LoadedModel<?> loaded =
                Models.load(model, Arena.ofAuto(), Map.of("media", media), tokenizer);

        RecordingProvider.Call call = RecordingProvider.last();
        assertEquals("language", call.kind());
        assertEquals(model, call.path());
        assertEquals(Map.of("media", media), call.companions());
        assertSame(tokenizer, call.tokenizer());
        assertEquals("fake", call.gguf().getString("general.architecture"));
        assertEquals("T", loaded.chatTemplateSource());
        assertFalse(call.channel().isOpen(), "the channel is closed after the load");
    }

    @Test
    void companionsJoinTheCacheSeed() throws IOException {
        Path model = write(fake(), "m.gguf");
        Path media = Files.write(dir.resolve("mmproj.gguf"), new byte[] {1, 2, 3});

        LoadedModel<?> bare = Models.load(model, Arena.ofAuto());
        LoadedModel<?> withMedia = Models.load(model, Arena.ofAuto(), Map.of("media", media));

        assertEquals(bare.seed(), Models.load(model, Arena.ofAuto()).seed());
        assertNotEquals(bare.seed(), withMedia.seed());
    }

    @Test
    void anUnofferedCompanionIsRefusedBeforeThePortIsCalled() throws IOException {
        Path model = write(fake(), "m.gguf");
        Path draft = Files.write(dir.resolve("mtp.gguf"), new byte[] {1});

        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Models.load(model, Arena.ofAuto(), Map.of("speculation", draft)));

        assertTrue(
                refused.getMessage().contains("'fake' has no 'speculation'"), refused.getMessage());
        assertTrue(refused.getMessage().contains("lexicon"), refused.getMessage());
        assertTrue(refused.getMessage().contains("media"), refused.getMessage());
        assertNull(RecordingProvider.last());
    }

    @Test
    void aTokenizerOutsideTheModelsIdSpaceIsRefusedBeforeThePortIsCalled() throws IOException {
        Path model =
                write(
                        fake().putArrayOfString("tokenizer.ggml.tokens", new String[] {"a", "b"}),
                        "m.gguf");

        assertDoesNotThrow(
                () -> Models.load(model, Arena.ofAuto(), Map.of(), RecordingProvider.tokenizer(2)));
        assertNotNull(RecordingProvider.last());
        RecordingProvider.reset();

        assertThrows(
                IllegalArgumentException.class,
                () -> Models.load(model, Arena.ofAuto(), Map.of(), RecordingProvider.tokenizer(3)));
        assertNull(RecordingProvider.last());
        assertThrows(
                IllegalArgumentException.class,
                () -> Models.loadReranker(model, Arena.ofAuto(), RecordingProvider.tokenizer(3)));
        assertNull(RecordingProvider.last());
    }

    @Test
    void aKindThePortDoesNotImplementIsRefusedByName() throws IOException {
        Path model = write(fake(), "m.gguf");

        UnsupportedOperationException refused =
                assertThrows(
                        UnsupportedOperationException.class,
                        () -> Models.loadEmbedder(model, Arena.ofAuto()));

        assertEquals("'fake' is not an embedding architecture", refused.getMessage());
        assertNull(RecordingProvider.last());
    }

    @Test
    void rerankerLoadHandsThePortThePath() throws IOException {
        Path model = write(fake(), "reranker.gguf");

        LoadedReranker<?> loaded = Models.loadReranker(model, Arena.ofAuto());

        assertEquals("reranker", RecordingProvider.last().kind());
        assertEquals(model, RecordingProvider.last().path());
        assertEquals("reranker.gguf", loaded.name());
    }

    @Test
    void speechLoadValidatesCompanionsLikeAnyOther() throws IOException {
        Path model = write(fake(), "voice.gguf");
        Path lexicon = Files.write(dir.resolve("lexicon.bin"), new byte[] {7});

        Models.loadSpeech(model, Arena.ofAuto(), Map.of("lexicon", lexicon));
        RecordingProvider.Call call = RecordingProvider.last();
        assertEquals("speech", call.kind());
        assertEquals(model, call.path());
        assertEquals(Map.of("lexicon", lexicon), call.companions());

        RecordingProvider.reset();
        assertThrows(
                IllegalArgumentException.class,
                () -> Models.loadSpeech(model, Arena.ofAuto(), Map.of("voice", lexicon)));
        assertNull(RecordingProvider.last());
    }

    @Test
    void anEmbeddedGgufArrivesRelocatedWithTheArchivesPath() throws IOException {
        Path standalone =
                write(
                        fake().putTensor(TensorEntry.create("w", new long[] {2}, GGMLType.F32, 0)),
                        "standalone.gguf");
        byte[] header = Files.readAllBytes(standalone);
        long dataOffset = GGUF.read(standalone).getTensorDataOffset();
        byte[] payload =
                ByteBuffer.allocate(2 * Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .putFloat(1f)
                        .putFloat(2f)
                        .array();
        int base = 1024;
        byte[] bytes = new byte[base + (int) dataOffset + payload.length];
        System.arraycopy(header, 0, bytes, base, header.length);
        System.arraycopy(payload, 0, bytes, base + (int) dataOffset, payload.length);
        Path archive = Files.write(dir.resolve("executable"), bytes);
        long entrySize = bytes.length - base;

        try (FileChannel channel = FileChannel.open(archive, StandardOpenOption.READ)) {
            channel.position(base);
            GGUF entry = GGUF.read(channel);

            Models.loadSpeech(channel, entry, base, entrySize, archive, Arena.ofAuto(), Map.of());

            RecordingProvider.Call call = RecordingProvider.last();
            assertEquals("speech", call.kind());
            assertSame(channel, call.channel());
            assertEquals(archive, call.path());
            assertEquals(base + dataOffset, call.gguf().getTensorDataOffset());
            assertEquals(base + dataOffset, call.gguf().absoluteOffset(call.gguf().getTensor("w")));
            assertEquals("fake", call.gguf().getString("general.architecture"));

            // the same entry declared shorter than its tensors need: refused before the port
            RecordingProvider.reset();
            IllegalArgumentException truncated =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    Models.loadSpeech(
                                            channel,
                                            entry,
                                            base,
                                            header.length,
                                            archive,
                                            Arena.ofAuto(),
                                            Map.of()));
            assertTrue(truncated.getMessage().contains("tensor w"), truncated.getMessage());
            assertNull(RecordingProvider.last());
        }
    }

    @Test
    void unknownArchitectureNamesWhatTheClasspathOffers() throws IOException {
        Path model =
                write(Builder.newBuilder().putString("general.architecture", "nope"), "m.gguf");

        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class, () -> Models.load(model, Arena.ofAuto()));

        assertTrue(
                refused.getMessage().contains("no provider for architecture 'nope'"),
                refused.getMessage());
        assertTrue(refused.getMessage().contains("fake"), refused.getMessage());
        assertNull(RecordingProvider.last());
    }

    @Test
    void notAGgufAndMissingFilesAreRefusedUpFront() throws IOException {
        Path text = Files.writeString(dir.resolve("weights.safetensors"), "{}");

        IllegalArgumentException notGguf =
                assertThrows(
                        IllegalArgumentException.class, () -> Models.load(text, Arena.ofAuto()));
        assertTrue(notGguf.getMessage().contains("not a GGUF model file"), notGguf.getMessage());
        assertThrows(
                NoSuchFileException.class,
                () -> Models.load(dir.resolve("absent.gguf"), Arena.ofAuto()));
        assertNull(RecordingProvider.last());
    }

    @Test
    void splitGgufsAreRefusedUpFront() throws IOException {
        Path part =
                write(
                        fake().putLong("split.count", 3).putLong("split.no", 1),
                        "m-00002-of-00003.gguf");

        UnsupportedOperationException refused =
                assertThrows(
                        UnsupportedOperationException.class,
                        () -> Models.load(part, Arena.ofAuto()));

        assertTrue(refused.getMessage().contains("part 2 of a 3-file split"), refused.getMessage());
        assertNull(RecordingProvider.last());
    }

    @Test
    void companionFilesComeFromTheSelectedPort() throws IOException {
        Path model = write(fake(), "m.gguf");

        assertEquals(Map.of("media", "mmproj", "lexicon", "lexicon"), Models.companionFiles(model));
        assertNull(RecordingProvider.last(), "asking what is offered loads nothing");
    }
}
