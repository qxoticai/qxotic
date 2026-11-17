package com.qxotic.jam.libjam;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * The jar ships a .sha256 beside every native library, the name NativeLoader extracts it under; a
 * library without one is never loaded. The build writes them (pom.xml, natives-sha256).
 */
class BundledSha256Test {

    @Test
    void everyBundledLibraryHasItsSha256() throws Exception {
        Path dir = Path.of(NativeLoader.class.getResource("/com/qxotic/jam/native").toURI());
        List<Path> libs;
        try (Stream<Path> files = Files.walk(dir)) {
            libs = files.filter(p -> p.toString().matches(".*\\.(so|dylib|dll)")).toList();
        }
        assertFalse(libs.isEmpty(), "no library under " + dir);
        for (Path lib : libs) {
            byte[] sha = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(lib));
            assertEquals(
                    HexFormat.of().formatHex(sha),
                    Files.readString(Path.of(lib + ".sha256")).strip(),
                    lib.toString());
        }
    }
}
