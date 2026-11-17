package com.qxotic.jam.libjam;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Loads the bundled native {@code libjam} for the current OS/arch.
 *
 * <p>The fat {@code jam-native} jar carries one native library per platform under {@code
 * /com/qxotic/jam/native/<os>-<arch>/<libname>} (e.g. {@code linux-x86-64/libjam.so}, {@code
 * darwin-aarch64/libjam.dylib}, {@code windows-x86-64/jam.dll}). At first use this class detects
 * the platform, extracts the matching library to {@code <java.io.tmpdir>/<sha256>/<libname>} (the
 * digest from the {@code <libname>.sha256} beside it in the jar; an existing copy is reused), and
 * {@link System#load(String) loads} it.
 *
 * <p>Overrides, in order:
 *
 * <ol>
 *   <li>{@code -Djam.native.library.path=/abs/path/to/libjam.so} (or {@code
 *       JAM_NATIVE_LIBRARY_PATH}) - load that file directly;
 *   <li>a bundled resource for {@code <os>-<arch>} - extract &amp; load (the normal path);
 *   <li>fallback {@code System.loadLibrary("jam")} - for dev runs with the lib on {@code
 *       java.library.path}.
 * </ol>
 */
final class NativeLoader {
    private NativeLoader() {}

    private static boolean loaded;

    /** The library the user named explicitly, or "" to look for the bundled one. */
    static String override() {
        return config("jam.native.library.path", "");
    }

    static synchronized void load() {
        if (loaded) return;

        String override = override();
        if (!override.isEmpty()) {
            System.load(override);
            loaded = true;
            return;
        }

        String os = os(), arch = arch(), lib = System.mapLibraryName("jam"), res = resource();
        InputStream in = NativeLoader.class.getResourceAsStream(res);
        if (in == null) { // dev fallback: rely on java.library.path
            try {
                System.loadLibrary("jam");
                loaded = true;
                return;
            } catch (UnsatisfiedLinkError e) {
                throw new UnsatisfiedLinkError(
                        "jam: no bundled native library at "
                                + res
                                + ", and 'jam' is not on java.library.path (os="
                                + os
                                + " arch="
                                + arch
                                + ")");
            }
        }
        // One copy per library build, named by the .sha256 the build ships beside it: runs share
        // it,
        // and nothing piles up (Windows cannot delete a loaded DLL at exit).
        // ponytail: an existing copy is trusted; on a shared /tmp another user could plant one.
        try (InputStream src = in;
                InputStream sum = NativeLoader.class.getResourceAsStream(res + ".sha256")) {
            if (sum == null) throw new UnsatisfiedLinkError("jam: " + res + ".sha256 missing");
            String sha = new String(sum.readNBytes(64), StandardCharsets.US_ASCII);
            Path path = Path.of(System.getProperty("java.io.tmpdir"), sha, lib);
            if (!Files.exists(path)) {
                Files.createDirectories(path.getParent());
                // write aside, then rename: a concurrent run never sees a half-written file
                Path part = Files.createTempFile(path.getParent(), lib, ".part");
                Files.copy(src, part, StandardCopyOption.REPLACE_EXISTING);
                try {
                    Files.move(part, path, StandardCopyOption.ATOMIC_MOVE);
                } catch (IOException e) {
                    // Windows: another run's copy is in place (and loaded), so use it
                    Files.delete(part);
                }
            }
            try {
                System.load(path.toAbsolutePath().toString());
            } catch (UnsatisfiedLinkError e) {
                throw new UnsatisfiedLinkError(
                        e.getMessage()
                                + " (a temporary directory mounted noexec? set -Djava.io.tmpdir)");
            }
            loaded = true;
        } catch (IOException e) {
            throw new UnsatisfiedLinkError("jam: failed to extract " + res + ": " + e);
        }
    }

    /** Whether this jar carries a library for the current OS/arch. */
    static boolean bundled() {
        return NativeLoader.class.getResource(resource()) != null;
    }

    private static String resource() {
        // jam.dll / libjam.dylib / libjam.so
        return "/com/qxotic/jam/native/" + os() + "-" + arch() + "/" + System.mapLibraryName("jam");
    }

    /**
     * Resolve {@code -Dprop}, else its env form ({@code jam.x.y} → {@code JAM_X_Y}: upper-case,
     * dots→underscores), else {@code def}. The env form matches the native {@code JAM_*} vars;
     * shared by NativeJAM's binding selector.
     */
    static String config(String prop, String def) {
        String v = System.getProperty(prop);
        if (v == null || v.isEmpty())
            v = System.getenv(prop.toUpperCase(Locale.ROOT).replace('.', '_'));
        return (v == null || v.isEmpty()) ? def : v;
    }

    static String os() {
        String s = System.getProperty("os.name", "").toLowerCase();
        if (s.contains("win")) return "windows";
        if (s.contains("mac") || s.contains("darwin")) return "darwin";
        if (s.contains("linux")) return "linux";
        return s.replaceAll("[^a-z0-9]+", "");
    }

    static String arch() {
        String a = System.getProperty("os.arch", "").toLowerCase();
        switch (a) {
            case "amd64":
            case "x86_64":
            case "x64":
                return "x86-64";
            case "aarch64":
            case "arm64":
                return "aarch64";
            default:
                return a.replaceAll("[^a-z0-9]+", "");
        }
    }
}
