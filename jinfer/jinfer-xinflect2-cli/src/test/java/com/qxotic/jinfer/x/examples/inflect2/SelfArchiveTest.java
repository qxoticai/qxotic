package com.qxotic.jinfer.x.examples.inflect2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The overlay layout a native image relies on: executable bytes first, ZIP appended. What matters
 * is that {@code offset} points at the entry's real data, since tensors are mapped from there.
 */
class SelfArchiveTest {

    private static final byte[] PAYLOAD =
            "not really a gguf, but bytes are bytes".getBytes(StandardCharsets.UTF_8);
    private static final String EXECUTABLE = "#!/fake/executable\n";

    /** An "executable" with a ZIP holding one STORED entry appended to it. */
    private static Path selfArchive(Path directory) throws IOException {
        Path file = directory.resolve("fake-binary");
        var zipped = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(zipped)) {
            ZipEntry entry = new ZipEntry("models/tiny.gguf");
            entry.setMethod(ZipEntry.STORED); // mappable in place, which is the whole point
            entry.setSize(PAYLOAD.length);
            entry.setCompressedSize(PAYLOAD.length);
            CRC32 crc = new CRC32();
            crc.update(PAYLOAD);
            entry.setCrc(crc.getValue());
            zip.putNextEntry(entry);
            zip.write(PAYLOAD);
            zip.closeEntry();
        }
        try (OutputStream out = Files.newOutputStream(file)) {
            out.write(EXECUTABLE.getBytes(StandardCharsets.UTF_8));
            out.write(zipped.toByteArray());
        }
        return file;
    }

    @Test
    void entryOffsetPointsAtTheData(@TempDir Path directory) throws IOException {
        Path file = selfArchive(directory);
        try (SelfArchive archive = SelfArchive.open(file)) {
            SelfArchive.Entry entry = archive.entry("models/tiny.gguf");
            assertEquals(PAYLOAD.length, entry.size());
            assertTrue(
                    entry.offset() > EXECUTABLE.length(),
                    "data must start past the executable prefix");
            assertArrayEquals(PAYLOAD, archive.readAt(entry.offset(), entry.size()));
        }
    }

    @Test
    void listsUsableEntries(@TempDir Path directory) throws IOException {
        try (SelfArchive archive = SelfArchive.open(selfArchive(directory))) {
            List<SelfArchive.Entry> entries = archive.entries();
            assertEquals(1, entries.size());
            assertEquals("models/tiny.gguf", entries.get(0).name());
        }
    }

    @Test
    void missingEntryFails(@TempDir Path directory) throws IOException {
        try (SelfArchive archive = SelfArchive.open(selfArchive(directory))) {
            assertThrows(IOException.class, () -> archive.entry("nope.gguf"));
        }
    }

    /** A short read used to come back silently zero-padded, which would corrupt a GGUF header. */
    @Test
    void readingPastTheEndFails(@TempDir Path directory) throws IOException {
        Path file = selfArchive(directory);
        try (SelfArchive archive = SelfArchive.open(file)) {
            long size = Files.size(file);
            assertThrows(EOFException.class, () -> archive.readAt(size - 4, 64));
        }
    }
}
