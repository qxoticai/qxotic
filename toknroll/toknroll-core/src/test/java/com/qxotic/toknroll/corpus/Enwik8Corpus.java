package com.qxotic.toknroll.corpus;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Utility for managing and accessing the enwik8 corpus.
 *
 * <p>enwik8 is a 100MB file containing the first 10^9 bytes of Wikipedia XML. It is commonly used
 * as a benchmark corpus for text compression and tokenization.
 *
 * <p>This class handles downloading, caching, and chunking the corpus.
 */
public final class Enwik8Corpus {

    private static final String DOWNLOAD_URL = "https://www.mattmahoney.net/dc/enwik8.zip";
    private static final long EXPECTED_SIZE = 100_000_000L;
    private static final String CACHE_DIR =
            System.getProperty("user.home") + "/.cache/qxotic/tokenizers/corpus";

    private final Path enwik8Path;
    private final byte[] data;

    private Enwik8Corpus(byte[] data, Path path) {
        this.data = data;
        this.enwik8Path = path;
    }

    /**
     * Load or download the enwik8 corpus.
     *
     * @return The loaded corpus
     */
    public static Enwik8Corpus load() {
        Path cacheDir = Paths.get(CACHE_DIR);
        Path enwik8Path = cacheDir.resolve("enwik8");

        if (Files.exists(enwik8Path) && enwik8Path.toFile().length() == EXPECTED_SIZE) {
            try {
                return new Enwik8Corpus(Files.readAllBytes(enwik8Path), enwik8Path);
            } catch (IOException e) {
                // Fall through to download
            }
        }

        try {
            downloadAndExtract(cacheDir);
            return new Enwik8Corpus(Files.readAllBytes(enwik8Path), enwik8Path);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException("Failed to download enwik8 corpus", e);
        }
    }

    /**
     * Create chunks of the corpus for testing.
     *
     * @param chunkSizes The sizes of chunks to generate
     * @param samplesPerSize Number of samples per chunk size
     * @return List of chunks with offset, size, and text
     */
    public List<Chunk> generateChunks(List<Integer> chunkSizes, int samplesPerSize) {
        List<Chunk> chunks = new ArrayList<Chunk>();
        int totalSize = data.length;

        for (int size : chunkSizes) {
            int stride = Math.max(1, (totalSize - size) / samplesPerSize);
            for (int i = 0; i < samplesPerSize; i++) {
                int offset = Math.min(i * stride, totalSize - size);

                // Ensure we don't split UTF-8 sequences - find valid start
                while (offset > 0 && (data[offset] & 0xC0) == 0x80) {
                    offset--;
                }

                // Ensure we end at a valid UTF-8 boundary
                int end = offset + size;
                while (end < totalSize && (data[end] & 0xC0) == 0x80) {
                    end++;
                }

                byte[] chunkData = new byte[end - offset];
                System.arraycopy(data, offset, chunkData, 0, end - offset);
                String text = new String(chunkData, StandardCharsets.UTF_8);

                chunks.add(new Chunk(offset, end - offset, text));
            }
        }

        return chunks;
    }

    /**
     * Create chunks of the corpus using default parameters.
     *
     * @return List of chunks with various sizes
     */
    public List<Chunk> generateDefaultChunks() {
        return generateChunks(List.of(256, 1024, 4096, 16384), 20);
    }

    public byte[] data() {
        return data;
    }

    public Path path() {
        return enwik8Path;
    }

    public static final class Chunk {
        private final int offset;
        private final int size;
        private final String text;

        public Chunk(int offset, int size, String text) {
            this.offset = offset;
            this.size = size;
            this.text = text;
        }

        public int offset() {
            return offset;
        }

        public int size() {
            return size;
        }

        public String text() {
            return text;
        }

        @Override
        public String toString() {
            return String.format(
                    "Chunk[offset=%d, size=%d, text_length=%d]", offset, size, text.length());
        }
    }

    private static void downloadAndExtract(Path cacheDir) throws IOException, InterruptedException {
        cacheDir.toFile().mkdirs();
        Path zipPath = cacheDir.resolve("enwik8.zip");

        if (!Files.exists(zipPath)) {
            System.out.println("Downloading enwik8 from " + DOWNLOAD_URL + "...");
            HttpClient client =
                    HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
            HttpRequest request = HttpRequest.newBuilder().uri(URI.create(DOWNLOAD_URL)).build();

            HttpResponse<Path> response =
                    client.send(request, HttpResponse.BodyHandlers.ofFile(zipPath));

            if (response.statusCode() != 200) {
                throw new IOException("Failed to download enwik8: HTTP " + response.statusCode());
            }
            System.out.println("Downloaded enwik8.zip to " + zipPath);
        }

        System.out.println("Extracting enwik8.zip...");
        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(zipPath))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if ("enwik8".equals(entry.getName())) {
                    Path extracted = cacheDir.resolve("enwik8");
                    Files.copy(zis, extracted, StandardCopyOption.REPLACE_EXISTING);
                    System.out.println("Extracted enwik8 to " + extracted);
                    return;
                }
            }
        }

        throw new IOException("enwik8 file not found in zip archive");
    }
}
