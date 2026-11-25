package com.qxotic.jota.stablehlo;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;

/**
 * Locates the StableHLO sample files bundled under {@code src/test/resources/stablehlo-samples}.
 */
public final class Samples {

    private Samples() {}

    public static Path root() {
        URL url = Samples.class.getResource("/stablehlo-samples");
        if (url == null) {
            throw new IllegalStateException("stablehlo-samples test resources are missing");
        }
        try {
            return Path.of(url.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
    }

    /** Files of {@code dir} (relative to the samples root) whose name ends with {@code suffix}. */
    public static List<Path> files(String dir, String suffix) {
        return list(root().resolve(dir), suffix);
    }

    public static List<Path> list(Path dir, String suffix) {
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(p -> p.getFileName().toString().endsWith(suffix)).sorted().toList();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static String read(Path file) {
        try {
            return Files.readString(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** lit chunks that check diagnostics, or that contain nothing but comments. */
    public static boolean isNegativeOrEmpty(String chunk) {
        return chunk.contains("expected-error")
                || chunk.lines().allMatch(l -> l.isBlank() || l.strip().startsWith("//"));
    }
}
