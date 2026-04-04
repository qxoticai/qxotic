package com.qxotic.toknroll.testkit;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

final class FixtureJsonLoader {

    private FixtureJsonLoader() {}

    @SuppressWarnings("unchecked")
    static Map<String, Object> loadMap(Class<?> owner, String resourceName, String failureContext) {
        try (InputStream is = owner.getClassLoader().getResourceAsStream(resourceName)) {
            if (is == null) {
                throw new IllegalStateException("Missing " + resourceName);
            }
            String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            return (Map<String, Object>) Json.parse(json);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load " + failureContext, e);
        }
    }
}
