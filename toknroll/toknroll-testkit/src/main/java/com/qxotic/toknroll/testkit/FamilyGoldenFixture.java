package com.qxotic.toknroll.testkit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed reader for {@code ground_truth_model_families.json}.
 *
 * <p>If the fixture is not available on the test classpath, regenerate it with {@code python
 * benchmarks/generate_ground_truth.py}.
 */
public final class FamilyGoldenFixture {

    private final Map<String, Family> families;

    private FamilyGoldenFixture(Map<String, Family> families) {
        this.families = families;
    }

    public static FamilyGoldenFixture load() {
        Map<String, Object> root = loadRawJson();
        Object familiesObj = root.get("families");
        if (!(familiesObj instanceof Map<?, ?>)) {
            return new FamilyGoldenFixture(Collections.emptyMap());
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> familiesMap = (Map<String, Object>) familiesObj;
        Map<String, Family> parsed = new LinkedHashMap<>();

        for (Map.Entry<String, Object> familyEntry : familiesMap.entrySet()) {
            if (!(familyEntry.getValue() instanceof Map<?, ?>)) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> familyMap = (Map<String, Object>) familyEntry.getValue();
            String modelRef = asString(familyMap.get("model_ref"));
            String revision = asString(familyMap.get("revision"));

            Object casesObj = familyMap.get("cases");
            List<CaseData> cases = new ArrayList<>();
            if (casesObj instanceof Map<?, ?>) {
                @SuppressWarnings("unchecked")
                Map<String, Object> caseMap = (Map<String, Object>) casesObj;
                for (Map.Entry<String, Object> caseEntry : caseMap.entrySet()) {
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
                    String decoded = asString(c.get("decoded"));
                    int tokenCount =
                            c.get("token_count") instanceof Number
                                    ? ((Number) c.get("token_count")).intValue()
                                    : tokens.length;
                    cases.add(
                            new CaseData(
                                    caseEntry.getKey(),
                                    text,
                                    decoded != null ? decoded : text,
                                    tokens,
                                    tokenCount));
                }
            }

            parsed.put(familyEntry.getKey(), new Family(modelRef, revision, List.copyOf(cases)));
        }

        return new FamilyGoldenFixture(Collections.unmodifiableMap(parsed));
    }

    public List<CaseData> getCases(String familyId) {
        Family family = families.get(familyId);
        return family == null ? Collections.emptyList() : family.cases();
    }

    public List<CaseData> getSampledCases(String familyId, int maxCases) {
        List<CaseData> cases = getCases(familyId);
        int limit = Math.max(0, Math.min(maxCases, cases.size()));
        return List.copyOf(cases.subList(0, limit));
    }

    public Map<String, Family> families() {
        return families;
    }

    public static final class Family {
        private final String modelRef;
        private final String revision;
        private final List<CaseData> cases;

        public Family(String modelRef, String revision, List<CaseData> cases) {
            this.modelRef = modelRef;
            this.revision = revision;
            this.cases = cases;
        }

        public String modelRef() {
            return modelRef;
        }

        public String revision() {
            return revision;
        }

        public List<CaseData> cases() {
            return cases;
        }
    }

    public static final class CaseData {
        private final String caseId;
        private final String text;
        private final String decoded;
        private final int[] tokens;
        private final int tokenCount;

        public CaseData(String caseId, String text, String decoded, int[] tokens, int tokenCount) {
            this.caseId = caseId;
            this.text = text;
            this.decoded = decoded;
            this.tokens = tokens;
            this.tokenCount = tokenCount;
        }

        public String caseId() {
            return caseId;
        }

        public String text() {
            return text;
        }

        public String decoded() {
            return decoded;
        }

        public int[] tokens() {
            return tokens;
        }

        public int tokenCount() {
            return tokenCount;
        }
    }

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
                FamilyGoldenFixture.class,
                "ground_truth_model_families.json",
                "family golden fixture");
    }
}
