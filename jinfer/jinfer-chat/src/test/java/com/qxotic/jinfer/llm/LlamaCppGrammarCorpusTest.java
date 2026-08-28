package com.qxotic.jinfer.llm;

import static com.qxotic.jinfer.llm.GrammarSpecTest.BV;
import static com.qxotic.jinfer.llm.GrammarSpecTest.accepts;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

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
                    // --- keywords documented as ignored: permissive where llama.cpp constrains ---
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
                    Map.entry("string w/ min length 1", "string length bounds"),
                    Map.entry("string w/ min length 3", "string length bounds"),
                    Map.entry("string w/ max length", "string length bounds"),
                    Map.entry("string w/ min & max length", "string length bounds"),
                    Map.entry("simple pattern", "pattern"),
                    Map.entry("pattern with escapes", "pattern"),
                    Map.entry("min+max items", "item counts"),
                    // ordering now matches; what is left in this case is tags:[] vs minItems 1
                    Map.entry("required props", "item counts"),
                    Map.entry("exotic formats (list)", "format"),
                    Map.entry(
                            "additional properties can't override other properties",
                            "additionalProperties"),
                    Map.entry(
                            "object properties, additionalProperties: true",
                            "additionalProperties"),

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

    @Test
    void llamaCppCorpus() throws IOException {
        List<?> cases = (List<?>) Json.parse(resource("/llama-cpp/grammar-corpus.json"));
        var failures = new TreeMap<String, List<String>>();
        int checks = 0, ran = 0;

        for (Object o : cases) {
            Map<?, ?> c = (Map<?, ?>) o;
            String kind = String.valueOf(c.get("kind"));
            String desc = String.valueOf(c.get("desc"));
            if (DIVERGENCES.containsKey(desc)) continue;
            ran++;

            Grammar.Spec spec;
            try {
                spec =
                        "test_grammar".equals(kind)
                                ? Grammar.of(String.valueOf(c.get("src")), BV)
                                : Grammar.fromSchema(schema(String.valueOf(c.get("src"))), BV);
            } catch (RuntimeException e) {
                failures.computeIfAbsent(desc, k -> new ArrayList<>()).add("did not compile: " + e);
                continue;
            }

            for (Object s : (List<?>) c.get("passing")) {
                checks++;
                if (!accepts(spec, BV, String.valueOf(s)))
                    failures.computeIfAbsent(desc, k -> new ArrayList<>())
                            .add("must ACCEPT " + show(String.valueOf(s)));
            }
            for (Object s : (List<?>) c.get("failing")) {
                checks++;
                if (accepts(spec, BV, String.valueOf(s)))
                    failures.computeIfAbsent(desc, k -> new ArrayList<>())
                            .add("must REJECT " + show(String.valueOf(s)));
            }
        }

        System.out.println(
                "\nllama.cpp corpus: "
                        + ran
                        + " cases, "
                        + checks
                        + " checks, "
                        + failures.size()
                        + " cases with failures ("
                        + DIVERGENCES.size()
                        + " skipped, see DIVERGENCES)");
        if (!failures.isEmpty()) {
            StringBuilder sb = new StringBuilder("llama.cpp grammar corpus:\n");
            failures.forEach(
                    (name, msgs) -> {
                        sb.append("  ").append(name).append('\n');
                        msgs.forEach(m -> sb.append("      ").append(m).append('\n'));
                    });
            throw new AssertionError(sb.toString());
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
