package com.qxotic.toknroll.testkit;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed reader for {@code ground_truth_tokens.json}: tiktoken's own {@code text -> tokens} for
 * r50k, cl100k and o200k, fetched by {@code make test-fixtures} into {@code test-fixtures/tiktoken}
 * from the qxoticai/assets repository. The file carries text and tokens only; the decoded text is
 * the text (tiktoken round-trips), its bytes are its UTF-8 and the count is the token count - a
 * test that wants those asserts the tokenizer reproduces them. Regenerate with {@code python3
 * toknroll-benchmarks/generate_ground_truth.py --skip-model-families}.
 */
public final class TiktokenGoldenFixture {

    private final Map<String, List<CaseData>> encodings;

    private TiktokenGoldenFixture(Map<String, List<CaseData>> encodings) {
        this.encodings = encodings;
    }

    public static TiktokenGoldenFixture load() {
        Map<String, Object> root = loadRawJson();
        Map<String, List<CaseData>> parsed = new LinkedHashMap<>();
        for (Map.Entry<String, Object> encodingEntry : root.entrySet()) {
            if (!(encodingEntry.getValue() instanceof Map<?, ?>)) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> casesMap = (Map<String, Object>) encodingEntry.getValue();
            List<CaseData> cases = new ArrayList<>();
            for (Map.Entry<String, Object> caseEntry : casesMap.entrySet()) {
                if (!(caseEntry.getValue() instanceof Map<?, ?>)) {
                    continue;
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> c = (Map<String, Object>) caseEntry.getValue();
                String text = asString(c.get("text"));
                List<Object> tokenValues = asList(c.get("tokens"));
                if (text == null || tokenValues == null) {
                    continue;
                }
                int[] tokens = toIntArray(tokenValues);
                cases.add(
                        new CaseData(
                                caseEntry.getKey(),
                                text,
                                text,
                                tokens,
                                text.getBytes(StandardCharsets.UTF_8),
                                tokens.length));
            }
            parsed.put(encodingEntry.getKey(), Collections.unmodifiableList(cases));
        }
        return new TiktokenGoldenFixture(Collections.unmodifiableMap(parsed));
    }

    public List<CaseData> getCases(String encoding) {
        List<CaseData> cases = encodings.get(encoding);
        return cases == null ? Collections.emptyList() : cases;
    }

    public List<CaseData> getSampledCases(String encoding, int maxCases) {
        List<CaseData> cases = getCases(encoding);
        int limit = Math.max(0, Math.min(maxCases, cases.size()));
        return List.copyOf(cases.subList(0, limit));
    }

    public record CaseData(
            String caseId,
            String inputText,
            String decoded,
            int[] tokens,
            byte[] decodedBytes,
            int tokenCount) {}

    private static String asString(Object value) {
        return value instanceof String ? (String) value : null;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(Object value) {
        return value instanceof List<?> ? (List<Object>) value : null;
    }

    private static int[] toIntArray(List<Object> values) {
        int[] arr = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            arr[i] = ((Number) values.get(i)).intValue();
        }
        return arr;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> loadRawJson() {
        return FixtureJsonLoader.loadMap(
                TiktokenGoldenFixture.class, "tiktoken/ground_truth_tokens.json", "golden fixture");
    }
}
