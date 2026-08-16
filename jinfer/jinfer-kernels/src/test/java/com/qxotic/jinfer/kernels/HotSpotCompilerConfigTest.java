package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/** Keeps the external HotSpot inline directives attached to real classes and methods. */
class HotSpotCompilerConfigTest {

    @Test
    void everyInlineTargetExists() throws Exception {
        Path config = findConfig();
        assertNotNull(config, "hotspot_compiler not found above the working directory");

        for (String line : Files.readAllLines(config)) {
            if (!line.startsWith("inline ")) continue;
            String target = line.substring("inline ".length()).trim();
            int separator = target.lastIndexOf('.');
            assertTrue(separator > 0, "malformed inline directive: " + line);

            String className = target.substring(0, separator).replace('/', '.');
            String methodName = target.substring(separator + 1);
            Method[] methods = Class.forName(className).getDeclaredMethods();
            assertTrue(
                    Arrays.stream(methods).anyMatch(method -> method.getName().equals(methodName)),
                    "missing HotSpot inline target " + target);
        }
    }

    private static Path findConfig() {
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent()) {
            Path candidate = dir.resolve("hotspot_compiler");
            if (Files.isRegularFile(candidate)) return candidate;
        }
        return null;
    }
}
