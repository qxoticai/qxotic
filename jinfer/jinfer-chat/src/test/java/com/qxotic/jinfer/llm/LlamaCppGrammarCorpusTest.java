package com.qxotic.jinfer.llm;

import static com.qxotic.jinfer.llm.GrammarMembership.BV;
import static com.qxotic.jinfer.llm.GrammarMembership.accepts;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * llama.cpp's grammar corpus, run against this engine. The cases are lifted verbatim from {@code
 * tests/test-grammar-integration.cpp} (MIT): each is a GBNF grammar or a JSON Schema plus the
 * strings it must accept and reject. GBNF is a SHARED format and acceptance is a
 * spelling-independent property, so the corpus transfers even though the two compilers name their
 * rules differently and llama.cpp's expected-grammar text does not transfer at all.
 *
 * <p>Every case runs, none is skipped. A schema keyword this compiler cannot enforce must be
 * REFUSED, naming it ({@link #REFUSED}); where llama.cpp is narrower than the JSON and JSON Schema
 * specs, the spec's answer is asserted instead, with the reason ({@link #SPEC_OVER_LLAMA}). Any
 * other disagreement is a failure.
 */
final class LlamaCppGrammarCorpusTest {

    /**
     * Cases whose schema carries a keyword a grammar cannot enforce: refused, the keyword named.
     */
    private static final Map<String, String> REFUSED =
            Map.ofEntries(
                    Map.entry("simple pattern", "pattern"),
                    Map.entry("pattern with escapes", "pattern"),
                    Map.entry(
                            "additional properties can't override other properties",
                            "additionalProperties"),
                    Map.entry(
                            "object properties, additionalProperties: true",
                            "additionalProperties"),
                    // the JSON Schema site's Product example: price > 0, unique tags
                    Map.entry("required props", "exclusiveMinimum"));

    /**
     * Strings llama.cpp rejects that the specs accept, so this engine accepts them: llama.cpp is
     * narrower than the spec there, not this engine wider.
     */
    private static final Map<String, Set<String>> SPEC_OVER_LLAMA =
            Map.of(
                    // JSON Schema: {} is any JSON value; llama.cpp reads it as any object
                    "empty schema (object)", Set.of("[]", "null", "\"\"", "true"),
                    // JSON integers have no digit limit and JSON whitespace is free; llama.cpp
                    // caps an integer at 16 digits, bounded or not, and its trailing whitespace at
                    // one space
                    "integer", Set.of("12345678901234567  "),
                    "min 0", Set.of("100000000000000000000000000000000"),
                    "min 2", Set.of("12345678900000000"));

    static Stream<Arguments> cases() throws IOException {
        List<?> cases = (List<?>) Json.parse(resource("/llama-cpp/grammar-corpus.json"));
        return cases.stream()
                .map(o -> (Map<?, ?>) o)
                .map(c -> Arguments.of(String.valueOf(c.get("desc")), c));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void llamaCppCorpus(String description, Map<?, ?> c) {
        String src = String.valueOf(c.get("src"));
        boolean gbnf = "test_grammar".equals(c.get("kind"));
        String keyword = REFUSED.get(description);
        if (keyword != null) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Grammar.fromSchema(schema(src), BV));
            assertTrue(e.getMessage().contains("\"" + keyword + "\""), e.getMessage());
            return;
        }
        Grammar.Spec spec = gbnf ? Grammar.of(src, BV) : Grammar.fromSchema(schema(src), BV);
        Set<String> specAccepts = SPEC_OVER_LLAMA.getOrDefault(description, Set.of());
        for (Object s : (List<?>) c.get("passing"))
            assertTrue(
                    accepts(spec, BV, String.valueOf(s)),
                    () -> "must ACCEPT " + show(String.valueOf(s)));
        for (Object s : (List<?>) c.get("failing")) {
            boolean accept = specAccepts.contains(String.valueOf(s));
            assertEquals(
                    accept,
                    accepts(spec, BV, String.valueOf(s)),
                    () ->
                            (accept ? "must ACCEPT, as the spec does: " : "must REJECT ")
                                    + show(String.valueOf(s)));
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> schema(String json) {
        return (Map<String, Object>) Json.parse(json);
    }

    private static String resource(String path) throws IOException {
        try (InputStream in = LlamaCppGrammarCorpusTest.class.getResourceAsStream(path)) {
            if (in == null) throw new IOException("missing test resource " + path);
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static String show(String s) {
        String one = s.replace("\n", "\\n").replace("\t", "\\t");
        return '"' + (one.length() <= 60 ? one : one.substring(0, 60) + "...") + '"';
    }
}
