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
 * toknroll-benchmarks/generate_ground_truth.py}.
 */
public final class FamilyGoldenFixture {

    private final Map<String, Family> families;

    private FamilyGoldenFixture(Map<String, Family> families) {
        this.families = families;
    }

    /** Reads the golden; a family or case missing a field fails here, never drops out. */
    @SuppressWarnings("unchecked")
    public static FamilyGoldenFixture load() {
        Map<String, Object> families = (Map<String, Object>) loadRawJson().get("families");
        if (families == null) {
            throw new IllegalStateException("family golden fixture has no families");
        }
        Map<String, Family> parsed = new LinkedHashMap<>();
        for (Map.Entry<String, Object> familyEntry : families.entrySet()) {
            String familyId = familyEntry.getKey();
            Map<String, Object> family = (Map<String, Object>) familyEntry.getValue();
            Map<String, Object> caseMap = (Map<String, Object>) require(family, "cases", familyId);
            List<CaseData> cases = new ArrayList<>();
            for (Map.Entry<String, Object> caseEntry : caseMap.entrySet()) {
                String caseId = familyId + "/" + caseEntry.getKey();
                Map<String, Object> c = (Map<String, Object>) caseEntry.getValue();
                String text = (String) require(c, "text", caseId);
                int[] tokens = toIntArray((List<Object>) require(c, "tokens", caseId));
                String decoded = (String) c.get("decoded");
                cases.add(
                        new CaseData(
                                caseEntry.getKey(),
                                text,
                                decoded != null ? decoded : text,
                                tokens,
                                (String) c.get("known")));
            }
            parsed.put(
                    familyId,
                    new Family(
                            (String) require(family, "model_ref", familyId),
                            (String) family.get("revision"),
                            List.copyOf(cases)));
        }
        return new FamilyGoldenFixture(Collections.unmodifiableMap(parsed));
    }

    private static Object require(Map<String, Object> map, String key, String where) {
        Object value = map.get(key);
        if (value == null) {
            throw new IllegalStateException("family golden fixture: " + where + " has no " + key);
        }
        return value;
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
        private final String known;

        /** {@code known} names the cause of a divergence already diagnosed, or is null. */
        public CaseData(String caseId, String text, String decoded, int[] tokens, String known) {
            this.caseId = caseId;
            this.text = text;
            this.decoded = decoded;
            this.tokens = tokens;
            this.known = known;
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

        public String known() {
            return known;
        }
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
