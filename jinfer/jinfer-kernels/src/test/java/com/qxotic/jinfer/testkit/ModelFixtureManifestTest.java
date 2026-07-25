package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * {@link ModelFixture} and {@code scripts/models.txt} are two views of one list - the fixture is
 * what tests resolve, the manifest is what the download script fetches. This gate keeps them
 * identical so neither can drift silently.
 */
class ModelFixtureManifestTest {

    @Test
    void manifestMatchesFixture() throws Exception {
        Path manifest = null;
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent()) {
            Path candidate = dir.resolve("scripts/models.txt");
            if (Files.exists(candidate)) {
                manifest = candidate;
                break;
            }
        }
        Assumptions.assumeTrue(manifest != null, "scripts/models.txt not found above the cwd");
        List<String> manifestLines =
                Files.readAllLines(manifest).stream()
                        .map(String::strip)
                        .filter(l -> !l.isEmpty() && !l.startsWith("#"))
                        .sorted()
                        .toList();
        List<String> fixtureLines =
                ModelFixture.all().stream()
                        .map(g -> g.source() + " " + g.user() + " " + g.repo() + " " + g.file())
                        .sorted()
                        .toList();
        assertEquals(fixtureLines, manifestLines, "scripts/models.txt drifted from ModelFixture");
    }
}
