package com.qxotic.jam.libjam;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jam.JAM;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Where the bundled library is unpacked, and what jam says when it cannot use it, as a user's
 * process sees it: each case starts a JVM of its own, with its own HOME and XDG_CACHE_HOME (the
 * Linux cache; macOS and Windows take their per-user directories, checked by hand on each).
 */
class NativeLoaderTest {

    @BeforeAll
    static void linuxOnly() {
        Assumptions.assumeTrue(NativeLoader.os().equals("linux"), "the XDG cache is Linux's");
    }

    @Test
    void theLibraryIsUnpackedOnceIntoTheCacheAndReused(@TempDir Path dir) throws Exception {
        Path cache = dir.resolve("xdg");
        Map<String, String> env =
                Map.of("HOME", dir.toString(), "XDG_CACHE_HOME", cache.toString());

        assertEquals("available=true", probe(classes(), env).strip());
        Path lib = theOnlyLibraryUnder(cache.resolve("jinfer/jam"));
        long written = Files.getLastModifiedTime(lib).toMillis();

        assertEquals("available=true", probe(classes(), env).strip());
        assertEquals(lib, theOnlyLibraryUnder(cache.resolve("jinfer/jam")));
        assertEquals(written, Files.getLastModifiedTime(lib).toMillis(), "rewritten, not reused");
    }

    @Test
    void aRelativeXdgCacheHomeIsIgnoredForTheHomeCache(@TempDir Path dir) throws Exception {
        Map<String, String> env = Map.of("HOME", dir.toString(), "XDG_CACHE_HOME", "relative");
        assertEquals("available=true", probe(classes(), env).strip());
        theOnlyLibraryUnder(dir.resolve(".cache/jinfer/jam"));
    }

    @Test
    void aLibraryWithoutItsSha256WarnsAndTheJavaBackendsTakeOver(@TempDir Path dir)
            throws Exception {
        Path copy = dir.resolve("classes");
        try (Stream<Path> files = Files.walk(classes())) {
            for (Path f : (Iterable<Path>) files::iterator) {
                Path to = copy.resolve(classes().relativize(f).toString());
                if (Files.isDirectory(f)) Files.createDirectories(to);
                else if (!f.toString().endsWith(".sha256")) Files.copy(f, to);
            }
        }
        Map<String, String> env =
                Map.of("HOME", dir.toString(), "XDG_CACHE_HOME", dir.resolve("xdg").toString());
        String out = probe(copy, env);
        assertTrue(out.contains("available=false"), out);
        assertTrue(out.contains("WARNING: jam: the native library did not load"), out);
        assertTrue(out.contains(".sha256 missing"), out);
    }

    /** jam-native's own classes and natives, as the build packages them. */
    private static Path classes() throws Exception {
        return Path.of(
                NativeLoader.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    }

    private static Path theOnlyLibraryUnder(Path root) throws IOException {
        try (Stream<Path> files = Files.walk(root)) {
            List<Path> libs =
                    files.filter(p -> p.getFileName().toString().equals("libjam.so")).toList();
            assertEquals(1, libs.size(), "libraries under " + root + ": " + libs);
            return libs.get(0);
        }
    }

    /** NativeLoaderProbe in a new JVM: jam-native from {@code jamNative}, its output and errors. */
    private static String probe(Path jamNative, Map<String, String> env) throws Exception {
        String classpath =
                String.join(
                        File.pathSeparator,
                        jamNative.toString(),
                        location(NativeLoaderProbe.class).toString(), // test classes
                        location(JAM.class).toString()); // jam-core
        String java = ProcessHandle.current().info().command().orElseThrow();
        ProcessBuilder pb =
                new ProcessBuilder(
                                java,
                                "--enable-native-access=ALL-UNNAMED",
                                "-cp",
                                classpath,
                                NativeLoaderProbe.class.getName())
                        .redirectErrorStream(true);
        pb.environment().remove("JAM_NATIVE_LIBRARY_PATH");
        // a JVM given options this way announces them ("Picked up JAVA_TOOL_OPTIONS: ...") in the
        // output compared here; the loader is what is under test, not the caller's JIT flags
        for (String options : List.of("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS"))
            pb.environment().remove(options);
        pb.environment().putAll(env);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes());
        assertEquals(0, p.waitFor(), out);
        return out;
    }

    private static Path location(Class<?> c) throws Exception {
        return Path.of(c.getProtectionDomain().getCodeSource().getLocation().toURI());
    }
}
