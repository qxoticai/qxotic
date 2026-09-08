package com.qxotic.jinfer.tts;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SelfArchiveTest {
    private static final byte[] PAYLOAD = "embedded model bytes".getBytes(StandardCharsets.UTF_8);

    @Test
    void readsAndExtractsStoredEntryFromExecutableOverlay(@TempDir Path directory)
            throws IOException {
        Path file = archive(directory, ZipEntry.STORED);
        try (SelfArchive archive = SelfArchive.open(file)) {
            SelfArchive.Entry entry = archive.entry("models/tiny.gguf");
            assertTrue(entry.offset() > 0);
            byte[] read;
            try (var input = Channels.newInputStream(archive.channel(entry))) {
                read = input.readAllBytes();
            }
            assertArrayEquals(PAYLOAD, read);
            Path extracted = archive.extract(entry);
            try {
                assertArrayEquals(PAYLOAD, Files.readAllBytes(extracted));
            } finally {
                Files.deleteIfExists(extracted);
            }
            assertEquals(1, archive.entries().size());
        }
    }

    @Test
    void rejectsCompressedEntries(@TempDir Path directory) throws IOException {
        try (SelfArchive archive = SelfArchive.open(archive(directory, ZipEntry.DEFLATED))) {
            assertThrows(IOException.class, () -> archive.entry("models/tiny.gguf"));
        }
    }

    private static Path archive(Path directory, int method) throws IOException {
        ByteArrayOutputStream zipped = new ByteArrayOutputStream();
        try (ZipOutputStream zip = new ZipOutputStream(zipped)) {
            ZipEntry entry = new ZipEntry("models/tiny.gguf");
            entry.setMethod(method);
            if (method == ZipEntry.STORED) {
                CRC32 crc = new CRC32();
                crc.update(PAYLOAD);
                entry.setSize(PAYLOAD.length);
                entry.setCompressedSize(PAYLOAD.length);
                entry.setCrc(crc.getValue());
            }
            zip.putNextEntry(entry);
            zip.write(PAYLOAD);
            zip.closeEntry();
        }
        Path file = directory.resolve("executable");
        try (OutputStream out = Files.newOutputStream(file)) {
            out.write("#!/fake\n".getBytes(StandardCharsets.US_ASCII));
            out.write(zipped.toByteArray());
        }
        return file;
    }
}
