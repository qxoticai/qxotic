package com.qxotic.toknroll.testkit.corpus;

import com.qxotic.toknroll.testkit.TestCachePaths;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/** The enwik8 corpus for tests, fetched once into the shared corpus cache. */
public final class Enwik8Corpus {

    private static final String DOWNLOAD_URL = "https://www.mattmahoney.net/dc/enwik8.zip";
    private static final long EXPECTED_SIZE = 100_000_000L;

    private final byte[] data;

    private Enwik8Corpus(byte[] data) {
        this.data = data;
    }

    public static Enwik8Corpus load() {
        Path dir = TestCachePaths.corpusDir();
        Path enwik8 = dir.resolve("enwik8");
        try {
            if (!Files.exists(enwik8) || Files.size(enwik8) != EXPECTED_SIZE) {
                download(dir, enwik8);
            }
            return new Enwik8Corpus(Files.readAllBytes(enwik8));
        } catch (IOException e) {
            throw new UncheckedIOException("enwik8 corpus at " + enwik8, e);
        }
    }

    public byte[] data() {
        return data;
    }

    private static void download(Path dir, Path enwik8) throws IOException {
        Files.createDirectories(dir);
        Path zip = dir.resolve("enwik8.zip");
        if (!Files.exists(zip)) {
            System.out.println("Downloading enwik8 from " + DOWNLOAD_URL + "...");
            HttpClient client =
                    HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
            HttpResponse<Path> response;
            try {
                response =
                        client.send(
                                HttpRequest.newBuilder(URI.create(DOWNLOAD_URL)).build(),
                                HttpResponse.BodyHandlers.ofFile(zip));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted downloading enwik8", e);
            }
            if (response.statusCode() != 200) {
                Files.deleteIfExists(zip);
                throw new IOException("Failed to download enwik8: HTTP " + response.statusCode());
            }
        }
        try (ZipInputStream in = new ZipInputStream(Files.newInputStream(zip))) {
            for (ZipEntry entry; (entry = in.getNextEntry()) != null; ) {
                if ("enwik8".equals(entry.getName())) {
                    Files.copy(in, enwik8, StandardCopyOption.REPLACE_EXISTING);
                    return;
                }
            }
        }
        throw new IOException("enwik8 not found in " + zip);
    }
}
