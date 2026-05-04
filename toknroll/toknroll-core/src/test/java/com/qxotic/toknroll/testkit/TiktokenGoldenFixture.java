package com.qxotic.toknroll.testkit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed reader for {@code ground_truth_tokens.json}.
 *
 * <p>If the fixture is not available on the test classpath, regenerate it with {@code python
 * toknroll-benchmarks/generate_ground_truth.py}.
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
                String decoded = asString(c.get("decoded"));
                List<Object> tokenValues = asList(c.get("tokens"));
                List<Object> decodedByteValues = asList(c.get("decoded_bytes"));
                if (decoded == null || tokenValues == null || decodedByteValues == null) {
                    continue;
                }
                int[] tokens = toIntArray(tokenValues);
                byte[] decodedBytes = toByteArray(decodedByteValues);
                int tokenCount =
                        c.get("token_count") instanceof Number
                                ? ((Number) c.get("token_count")).intValue()
                                : tokens.length;
                String inputText = chooseInputText(text, decoded);
                cases.add(
                        new CaseData(
                                caseEntry.getKey(),
                                inputText,
                                decoded,
                                tokens,
                                decodedBytes,
                                tokenCount));
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

    private static byte[] toByteArray(List<Object> values) {
        byte[] arr = new byte[values.size()];
        for (int i = 0; i < values.size(); i++) {
            arr[i] = ((Number) values.get(i)).byteValue();
        }
        return arr;
    }

    private static String chooseInputText(String text, String decoded) {
        if (text == null) {
            return decoded;
        }
        if (isAllQuestionMarks(text) && !isAllQuestionMarks(decoded)) {
            return decoded;
        }
        return text;
    }

    private static boolean isAllQuestionMarks(String value) {
        if (value == null || value.isEmpty()) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            if (value.charAt(i) != '?') {
                return false;
            }
        }
        return true;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> loadRawJson() {
        return FixtureJsonLoader.loadMap(
                TiktokenGoldenFixture.class, "ground_truth_tokens.json", "golden fixture");
    }
}
