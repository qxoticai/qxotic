package com.qxotic.jinfer.llm;

import static com.qxotic.jinfer.llm.GrammarMembership.BV;
import static com.qxotic.jinfer.llm.GrammarMembership.accepts;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
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
 * <p>Two things this pins that hand-written cases do not: an INDEPENDENT author's idea of what the
 * notation means (quantifiers, repetition bounds, escapes, char classes), and the JSON Schema
 * keywords a serious implementation is expected to honour. Where jinfer's documented subset is
 * narrower than llama.cpp's, or the two simply chose differently, the case is listed in {@link
 * #DIVERGENCES} with the reason - that list is the honest coverage report, not a set of excuses: a
 * keyword leaving it is a feature landing, and a case failing OUTSIDE it is a bug.
 */
final class LlamaCppGrammarCorpusTest {

    /**
     * Cases this engine does not agree with, each named with the divergence it exposes. Two kinds:
     * keywords {@link Grammar#fromSchema} documents as IGNORED (so the grammar is permissive where
     * llama.cpp is strict - it accepts documents llama.cpp rejects), and genuine semantic choices
     * that differ. This list IS the coverage report: a keyword leaving it is a feature landing, a
     * case failing outside it is a bug.
     */
    private static final Map<String, String> DIVERGENCES =
            Map.ofEntries(
                    // --- keywords documented as ignored: PERMISSIVE where llama.cpp constrains,
                    //     so this compiler accepts documents llama.cpp rejects ---
                    Map.entry("min 0", "numeric bounds"),
                    Map.entry("min 2", "numeric bounds"),
                    Map.entry("min 456", "numeric bounds"),
                    Map.entry("min -123", "numeric bounds"),
                    Map.entry("max 9999", "numeric bounds"),
                    Map.entry("max -9999", "numeric bounds"),
                    Map.entry("min 5 max 30", "numeric bounds"),
                    Map.entry("min 1 max 900719925474091", "numeric bounds"),
                    Map.entry("min -1 max 1", "numeric bounds"),
                    Map.entry("min -123 max 42", "numeric bounds"),
                    Map.entry("exclusive min / max", "numeric bounds"),
                    Map.entry("simple pattern", "pattern"),
                    Map.entry("pattern with escapes", "pattern"),

                    // --- STRICTER than llama.cpp, and deliberately left so ---

                    // additionalProperties. The object rule admits exactly the declared properties,
                    // so an undeclared key has no path: this compiler behaves as
                    // additionalProperties:false ALWAYS. That is right for the common case - a
                    // generated schema for a closed POJO says false - and safe when it is wrong,
                    // since the model is held to the declared fields rather than inventing some.
                    // Supporting `true` means spelling out "any string EXCEPT these", which GBNF
                    // cannot say directly: llama.cpp builds a trie of the declared names and emits
                    // its complement (~60 lines), then interleaves the extra pairs with the ordered
                    // subset already built here. Declined on purpose: highest cost of anything
                    // left,
                    // to support schemas that ask a CONSTRAINED decode to also accept anything
                    // else.
                    Map.entry(
                            "additional properties can't override other properties",
                            "additionalProperties (stricter: declared keys only)"),
                    Map.entry(
                            "object properties, additionalProperties: true",
                            "additionalProperties (stricter: declared keys only)"),

                    // --- semantic choices that differ ---

                    // An empty schema {} admits ANY JSON here (what the spec says); llama.cpp
                    // restricts it
                    // to objects. Their own case name says "(object)".
                    Map.entry("empty schema (object)", "empty schema means any JSON, not object"),

                    // Trailing whitespace: this engine's ws is [ \t\n\r]{0,8} anywhere, llama.cpp's
                    // space rule admits ONE space (or a newline plus indent), so it rejects two.
                    Map.entry("integer", "bounded-whitespace policy"),
                    // GBNF ".": one BYTE here, one UTF-8 CODE POINT in llama.cpp, so "... abc ..."
                    // matches
                    // three emoji there and three bytes of the first emoji here.
                    Map.entry("special characters", "dot matches a byte, not a code point"));

    static Stream<Arguments> cases() throws IOException {
        List<?> cases = (List<?>) Json.parse(resource("/llama-cpp/grammar-corpus.json"));
        return cases.stream()
                .map(o -> (Map<?, ?>) o)
                .map(c -> Arguments.of(String.valueOf(c.get("desc")), c));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void llamaCppCorpus(String description, Map<?, ?> c) {
        Assumptions.assumeFalse(DIVERGENCES.containsKey(description), DIVERGENCES.get(description));
        Grammar.Spec spec =
                "test_grammar".equals(c.get("kind"))
                        ? Grammar.of(String.valueOf(c.get("src")), BV)
                        : Grammar.fromSchema(schema(String.valueOf(c.get("src"))), BV);
        for (Object s : (List<?>) c.get("passing"))
            assertTrue(
                    accepts(spec, BV, String.valueOf(s)),
                    () -> "must ACCEPT " + show(String.valueOf(s)));
        for (Object s : (List<?>) c.get("failing"))
            assertTrue(
                    !accepts(spec, BV, String.valueOf(s)),
                    () -> "must REJECT " + show(String.valueOf(s)));
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
