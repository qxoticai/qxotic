package com.qxotic.toknroll.testkit;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

final class FixtureJsonLoader {

    private FixtureJsonLoader() {}

    @SuppressWarnings("unchecked")
    static Map<String, Object> loadMap(Class<?> owner, String resourceName, String failureContext) {
        try (InputStream is = owner.getClassLoader().getResourceAsStream(resourceName)) {
            if (is != null) {
                String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
                return (Map<String, Object>) Json.parse(json);
            }

            Path diskPath = fixturePathFromProperty(resourceName);
            if (Files.exists(diskPath)) {
                String json = Files.readString(diskPath, StandardCharsets.UTF_8);
                return (Map<String, Object>) Json.parse(json);
            }

            throw new IllegalStateException(missingFixtureMessage(resourceName, diskPath));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load " + failureContext, e);
        }
    }

    private static Path fixturePathFromProperty(String resourceName) {
        String root = System.getProperty("toknroll.test.fixtureDir", "src/test/resources");
        return Paths.get(root).resolve(resourceName);
    }

    private static String missingFixtureMessage(String resourceName, Path diskPath) {
        return "Missing fixture "
                + resourceName
                + " (also checked "
                + diskPath
                + "). Generate fixtures with: python3 benchmarks/generate-ground-truth.py";
    }
}
