package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The preload's matching policy: a hit must be PROVEN (size, then header digests), and every kind
 * of miss - wrong size, a size tie, same-size different-bytes - must fall back to null, never to a
 * wrong entry. The entries here carry no GGUF or tokenizer: match() must not need them.
 */
final class AOTTest {

    /** An entry as preload() would bake it for {@code file}, treating the whole file as header. */
    private static AOT.PreloadedFile entryFor(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            AOT.HeaderDigests digests = AOT.digestHeader(channel, channel.size());
            return new AOT.PreloadedFile(
                    file.getFileName().toString(),
                    channel.size(),
                    channel.size(),
                    digests.crc32c(),
                    digests.sha256(),
                    null,
                    null);
        }
    }

    @Test
    void aByteIdenticalFileMatchesWhateverItIsCalled(@TempDir Path dir) throws IOException {
        Path original = Files.writeString(dir.resolve("model.gguf"), "the same weights");
        AOT.PreloadedFile entry = entryFor(original);
        Path renamed = Files.copy(original, dir.resolve("renamed-elsewhere.bin"));

        assertSame(entry, AOT.match(List.of(entry), original));
        assertSame(entry, AOT.match(List.of(entry), renamed), "content is the identity, not name");
    }

    @Test
    void aDifferentSizeMisses(@TempDir Path dir) throws IOException {
        AOT.PreloadedFile entry = entryFor(Files.writeString(dir.resolve("model.gguf"), "weights"));
        Path swapped = Files.writeString(dir.resolve("model2.gguf"), "different length here");

        assertNull(AOT.match(List.of(entry), swapped));
    }

    @Test
    void sameSizeDifferentBytesIsCaughtByTheDigests(@TempDir Path dir) throws IOException {
        AOT.PreloadedFile entry =
                entryFor(Files.writeString(dir.resolve("model.gguf"), "the same weights"));
        Path impostor = Files.writeString(dir.resolve("impostor.gguf"), "the SAME weights");

        assertNotNull(
                AOT.match(List.of(entry), dir.resolve("model.gguf")), "sanity: real file hits");
        assertNull(
                AOT.match(List.of(entry), impostor),
                "equal size must not be enough - the header bytes differ");
    }

    @Test
    void aSizeTieMatchesNobody(@TempDir Path dir) throws IOException {
        Path a = Files.writeString(dir.resolve("a.gguf"), "sixteen bytes ok");
        Path b = Files.writeString(dir.resolve("b.gguf"), "also 16 bytes ok");
        List<AOT.PreloadedFile> tied = List.of(entryFor(a), entryFor(b));

        assertNull(AOT.match(tied, a), "ambiguity must never guess");
    }

    @Test
    void anUnreadableOrMissingFileMisses(@TempDir Path dir) throws IOException {
        AOT.PreloadedFile entry = entryFor(Files.writeString(dir.resolve("model.gguf"), "weights"));

        assertNull(AOT.match(List.of(entry), dir.resolve("no-such-file.gguf")));
    }

    @Test
    void preloadRejectsMissingFilesAndModelsWithoutVocabulary(@TempDir Path dir) throws Exception {
        assertEquals(List.of(), AOT.preload(""));
        assertThrows(
                IllegalArgumentException.class,
                () -> AOT.preload(dir.resolve("missing.gguf").toString()));
        Path file = dir.resolve("speech.gguf");
        GGUF.write(
                Builder.newBuilder().putString("general.architecture", "cli_test_speech").build(),
                file);
        var error =
                assertThrows(IllegalArgumentException.class, () -> AOT.preload(file.toString()));
        assertTrue(error.getMessage().contains("no vocabulary"));
    }

    @Test
    void shaMismatchAndTruncatedHeaderBothFallBack(@TempDir Path dir) throws Exception {
        Path file = Files.writeString(dir.resolve("model.gguf"), "weights");
        AOT.PreloadedFile actual = entryFor(file);
        var differentSha =
                new AOT.PreloadedFile(
                        actual.fileName(),
                        actual.fileSize(),
                        actual.headerLength(),
                        actual.headerCrc32c(),
                        "0".repeat(64),
                        null,
                        null);
        assertNull(
                AOT.match(List.of(differentSha), file),
                "CRC agreement alone must never select stale metadata");
        var truncated =
                new AOT.PreloadedFile(
                        actual.fileName(),
                        actual.fileSize(),
                        actual.headerLength() + 1,
                        actual.headerCrc32c(),
                        actual.headerSha256(),
                        null,
                        null);
        assertNull(AOT.match(List.of(truncated), file));
        Files.writeString(file, "new and longer weights");
        assertNull(AOT.match(List.of(actual), file), "same filename is not an identity");
    }
}
