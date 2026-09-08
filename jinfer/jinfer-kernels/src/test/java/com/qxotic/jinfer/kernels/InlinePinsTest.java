package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.jar.JarFile;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * The {@code -H:DirectedInline} pins shipped in {@code native-image.properties} must name methods
 * that exist, and only one module may ship them.
 *
 * <p>The pins hold open the calls a vector value crosses. When the AOT inliner declines one, the
 * returned vector cannot stay in a register, so it is materialized and every Vector API operation
 * in the caller de-intrinsifies into the generic per-lane path - all-or-nothing per method,
 * measured at 0.48 t/s against 27.5 t/s on LFM2.5-2.6B-Q8_0 decode with a GraalVM CE image.
 *
 * <p>Both ways of getting this wrong are SILENT, which is the only reason this test exists. A rule
 * whose caller or callee no longer exists is discarded with no build error and no warning, so a
 * rename costs 58x invisibly. And the option is single-valued rather than accumulating: when two
 * jars on the image classpath each set it, the last one wins and the other module's pins vanish
 * (measured - jam-vector's single rule displaced all of jinfer-kernels' and decode fell back to
 * 0.49 t/s). Neither failure shows up in a build log, a benchmark suite, or a unit test other than
 * this one.
 *
 * @see VectorSpeciesConstantTest for the sibling law, that a species must not cross a call either
 */
final class InlinePinsTest {

    private static final String OPTION = "-H:DirectedInline=";
    private static final String PREFIX = "META-INF/native-image/";
    private static final String FILE = "native-image.properties";

    @Test
    void everyInlinePinResolvesToARealMethod() {
        Map<String, List<String>> pins = shippedPins();
        assertFalse(
                pins.isEmpty(),
                "no "
                        + OPTION
                        + " rules on the classpath - if the kernels genuinely need no pins, delete"
                        + " this test rather than leaving it to pass vacuously");

        List<String> broken = new ArrayList<>();
        pins.forEach(
                (source, rules) -> {
                    for (String rule : rules) {
                        String[] endpoints = rule.split("->");
                        if (endpoints.length != 2) {
                            broken.add(rule + " is not caller->callee (from " + source + ")");
                            continue;
                        }
                        for (String endpoint : endpoints)
                            if (!resolves(endpoint.trim()))
                                broken.add(
                                        endpoint.trim()
                                                + " names no such method, in "
                                                + rule
                                                + " (from "
                                                + source
                                                + ")");
                    }
                });
        assertTrue(
                broken.isEmpty(),
                () ->
                        "native-image discards a pin that matches nothing, silently, so these ship"
                                + " de-intrinsified: "
                                + broken);
    }

    @Test
    void onlyOneModuleShipsInlinePins() {
        assertEquals(
                1,
                shippedPins().size(),
                () ->
                        OPTION
                                + " is single-valued, so the last module on the image classpath"
                                + " wins and every other module's pins are dropped without a"
                                + " warning. Move them into one file, or drop the call the other"
                                + " module was pinning (jam-vector writes its dequantize loop out"
                                + " for exactly this reason). Sources: "
                                + shippedPins().keySet());
    }

    /**
     * The pins each {@code native-image.properties} on the classpath ships, keyed by its module.
     */
    private static Map<String, List<String>> shippedPins() {
        Map<String, List<String>> pins = new LinkedHashMap<>();
        for (String entry : System.getProperty("java.class.path").split(File.pathSeparator))
            readPins(Path.of(entry), pins);
        return pins;
    }

    private static void readPins(Path entry, Map<String, List<String>> pins) {
        try {
            if (Files.isDirectory(entry)) {
                Path root = entry.resolve(PREFIX);
                if (!Files.isDirectory(root)) return;
                try (Stream<Path> tree = Files.walk(root)) {
                    for (Path file : tree.filter(p -> p.endsWith(FILE)).toList())
                        try (InputStream in = Files.newInputStream(file)) {
                            collect(entry.relativize(file).toString(), in, pins);
                        }
                }
            } else if (Files.isRegularFile(entry) && entry.toString().endsWith(".jar")) {
                try (JarFile jar = new JarFile(entry.toFile())) {
                    for (var element : jar.stream().toList()) {
                        String name = element.getName();
                        if (!name.startsWith(PREFIX) || !name.endsWith(FILE)) continue;
                        try (InputStream in = jar.getInputStream(element)) {
                            collect(name, in, pins);
                        }
                    }
                }
            }
        } catch (IOException e) {
            // An unreadable classpath entry is the build's problem, not this law's.
        }
    }

    private static void collect(String source, InputStream in, Map<String, List<String>> pins)
            throws IOException {
        var parsed = new Properties();
        parsed.load(in);
        for (String argument : parsed.getProperty("Args", "").split("\\s+"))
            if (argument.startsWith(OPTION))
                pins.computeIfAbsent(source, unused -> new ArrayList<>())
                        .addAll(Arrays.asList(argument.substring(OPTION.length()).split(",")));
    }

    /** Whether {@code com.pkg.Owner.method} names a method that is actually declared. */
    private static boolean resolves(String qualified) {
        int split = qualified.lastIndexOf('.');
        if (split < 0) return false;
        String method = qualified.substring(split + 1);
        try {
            return Arrays.stream(Class.forName(qualified.substring(0, split)).getDeclaredMethods())
                    .anyMatch(declared -> declared.getName().equals(method));
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            return false;
        }
    }
}
