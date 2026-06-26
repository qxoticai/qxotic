package com.qxotic.jam;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Loads the bundled native {@code libjam} for the current OS/arch.
 *
 * <p>The fat {@code jam.jar} carries one native library per platform under
 * {@code /com/qxotic/jam/native/<os>-<arch>/<libname>} (e.g. {@code linux-x86-64/libjam.so},
 * {@code darwin-aarch64/libjam.dylib}, {@code windows-x86-64/jam.dll}). At first use this class detects
 * the platform, extracts the matching library to a temp file, and {@link System#load(String) loads} it.
 *
 * <p>Overrides, in order:
 * <ol>
 *   <li>{@code -Djam.library.path=/abs/path/to/libjam.so} (or {@code JAM_LIBRARY_PATH}) — load that file directly;</li>
 *   <li>a bundled resource for {@code <os>-<arch>} — extract &amp; load (the normal path);</li>
 *   <li>fallback {@code System.loadLibrary("jam")} — for dev runs with the lib on {@code java.library.path}.</li>
 * </ol>
 */
final class NativeLoader {
    private NativeLoader() {}
    private static boolean loaded;

    static synchronized void load() {
        if (loaded) return;

        String override = config("jam.library.path", "");
        if (!override.isEmpty()) { System.load(override); loaded = true; return; }

        String os = os(), arch = arch(), lib = System.mapLibraryName("jam");   // jam.dll / libjam.dylib / libjam.so
        String res = "/com/qxotic/jam/native/" + os + "-" + arch + "/" + lib;
        InputStream in = NativeLoader.class.getResourceAsStream(res);
        if (in == null) {                                   // dev fallback: rely on java.library.path
            try { System.loadLibrary("jam"); loaded = true; return; }
            catch (UnsatisfiedLinkError e) {
                throw new UnsatisfiedLinkError("jam: no bundled native library at " + res
                    + ", and 'jam' is not on java.library.path (os=" + os + " arch=" + arch + ")");
            }
        }
        try (InputStream src = in) {
            Path dir = Files.createTempDirectory("jam-native");
            dir.toFile().deleteOnExit();
            Path tmp = dir.resolve(lib);
            Files.copy(src, tmp, StandardCopyOption.REPLACE_EXISTING);
            tmp.toFile().deleteOnExit();
            System.load(tmp.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw new UnsatisfiedLinkError("jam: failed to extract " + res + ": " + e);
        }
    }

    /** Resolve {@code -Dprop}, else its env form ({@code jam.x.y} → {@code JAM_X_Y}: upper-case, dots→underscores),
     *  else {@code def}. The env form matches the native {@code JAM_*} vars; shared by NativeJAM's binding selector. */
    static String config(String prop, String def) {
        String v = System.getProperty(prop);
        if (v == null || v.isEmpty()) v = System.getenv(prop.toUpperCase(Locale.ROOT).replace('.', '_'));
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
            case "amd64": case "x86_64": case "x64": return "x86-64";
            case "aarch64": case "arm64":            return "aarch64";
            default: return a.replaceAll("[^a-z0-9]+", "");
        }
    }

}
