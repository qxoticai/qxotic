package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * The invariant that makes borrow-vs-adopt mean anything: <b>a State allocates only from the arena
 * it was given</b>. Ownership is decided by the {@code newState} flavour and implemented once in
 * core, so a family cannot get the FREEING wrong - but a family that quietly allocated a buffer
 * from an arena of its own would leave memory alive after {@code close()}, and no ownership rule
 * would catch it. That is not hypothetical: {@code FlashAttention.DecodeScratch} was on {@code
 * ofAuto} in all seven families and survived every state close until it was found.
 *
 * <p>So: no arena creation anywhere inside a State class body. Towers, templates and loaders may
 * create arenas - they have their own lifetimes and their own reviews - which is why this scans
 * State bodies rather than whole files.
 */
final class StateArenaDisciplineTest {

    private static final Pattern STATE_CLASS =
            Pattern.compile("\\bclass\\s+(\\w*State)\\b[^{]*\\{");
    private static final Pattern CREATES_ARENA = Pattern.compile("Arena\\s*\\.\\s*(of\\w+|global)\\s*\\(");

    @Test
    void noStateCreatesItsOwnArena() throws IOException {
        List<Path> sources = jinferSources();
        Assumptions.assumeFalse(sources.isEmpty(), "sources not found above the working directory");

        List<String> offenders = new ArrayList<>();
        int scanned = 0;
        for (Path source : sources) {
            String text = Files.readString(source);
            Matcher declaration = STATE_CLASS.matcher(text);
            while (declaration.find()) {
                scanned++;
                String body = classBody(text, declaration.end() - 1);
                Matcher creation = CREATES_ARENA.matcher(body);
                if (creation.find()) {
                    offenders.add(
                            source.getFileName()
                                    + " -> class "
                                    + declaration.group(1)
                                    + " creates "
                                    + creation.group());
                }
            }
        }
        assertTrue(scanned > 5, "expected to scan several State classes, saw " + scanned);
        assertTrue(
                offenders.isEmpty(),
                "a State must allocate only from the arena it was given, so that close() frees"
                        + " everything it allocated:\n  "
                        + String.join("\n  ", offenders));
    }

    /** The source text between the brace at {@code open} and its match. */
    private static String classBody(String text, int open) {
        int depth = 0;
        for (int i = open; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '{') depth++;
            else if (c == '}' && --depth == 0) return text.substring(open, i + 1);
        }
        return text.substring(open);
    }

    private static List<Path> jinferSources() throws IOException {
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent()) {
            Path core = dir.resolve("jinfer-core/src/main/java");
            if (Files.isDirectory(core)) {
                try (Stream<Path> tree = Files.walk(dir)) {
                    return tree.filter(p -> p.toString().endsWith(".java"))
                            .filter(p -> p.toString().contains("/src/main/java/"))
                            .filter(p -> !p.toString().contains("/target/"))
                            .toList();
                }
            }
        }
        return List.of();
    }
}
