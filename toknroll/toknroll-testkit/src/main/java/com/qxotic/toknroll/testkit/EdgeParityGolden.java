package com.qxotic.toknroll.testkit;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/**
 * An edge parity golden ({@code hf_edge_parity.json}, {@code gguf_edge_parity.json}): the texts,
 * then each model's answer on every text, its ids and, for Hugging Face, the decoded text.
 */
public final class EdgeParityGolden {

    private final List<String> texts;
    private final List<Model> models;

    private EdgeParityGolden(List<String> texts, List<Model> models) {
        this.texts = texts;
        this.models = models;
    }

    /** Reads the golden from the test classpath; a missing or malformed one fails here. */
    @SuppressWarnings("unchecked")
    public static EdgeParityGolden load(Class<?> owner, String resource, String generator) {
        Map<String, Object> root =
                FixtureJsonLoader.loadMap(owner, resource, "edge parity golden", generator);
        List<String> texts = new ArrayList<>();
        for (Object text : (List<Object>) require(root, "texts", resource)) {
            texts.add((String) text);
        }
        List<Model> models = new ArrayList<>();
        for (Object m : (List<Object>) require(root, "models", resource)) {
            Map<String, Object> model = (Map<String, Object>) m;
            String name =
                    (String) (model.containsKey("ref") ? model.get("ref") : model.get("file"));
            if (name == null) {
                throw new IllegalStateException(resource + ": a model has neither ref nor file");
            }
            List<Object> cases = (List<Object>) require(model, "cases", resource + " " + name);
            if (cases.size() != texts.size()) {
                throw new IllegalStateException(
                        resource
                                + " "
                                + name
                                + ": "
                                + cases.size()
                                + " cases for "
                                + texts.size()
                                + " texts");
            }
            List<Expected> expected = new ArrayList<>();
            for (Object c : cases) {
                expected.add(expected(c, resource + " " + name));
            }
            models.add(
                    new Model(
                            name,
                            (String) model.get("revision"),
                            Collections.unmodifiableList(expected)));
        }
        return new EdgeParityGolden(
                Collections.unmodifiableList(texts), Collections.unmodifiableList(models));
    }

    @SuppressWarnings("unchecked")
    private static Expected expected(Object c, String where) {
        if (c instanceof List<?>) {
            return new Expected(toIntArray((List<Object>) c), null);
        }
        if (c instanceof Map<?, ?>) {
            Map<String, Object> map = (Map<String, Object>) c;
            return new Expected(
                    toIntArray((List<Object>) require(map, "ids", where)),
                    (String) require(map, "decoded", where));
        }
        throw new IllegalStateException(where + ": a case is " + c);
    }

    private static Object require(Map<String, Object> map, String key, String where) {
        Object value = map.get(key);
        if (value == null) {
            throw new IllegalStateException(where + " has no " + key);
        }
        return value;
    }

    private static int[] toIntArray(List<Object> values) {
        int[] ids = new int[values.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = ((Number) values.get(i)).intValue();
        }
        return ids;
    }

    public List<String> texts() {
        return texts;
    }

    public List<Model> models() {
        return models;
    }

    /** One model: its name (a hub ref or a file name), revision or null, and cases by text. */
    public static final class Model {
        private final String name;
        private final String revision;
        private final List<Expected> cases;

        Model(String name, String revision, List<Expected> cases) {
            this.name = name;
            this.revision = revision;
            this.cases = cases;
        }

        public String name() {
            return name;
        }

        public String revision() {
            return revision;
        }

        public List<Expected> cases() {
            return cases;
        }
    }

    /** The reference's answer on one text: its ids, and its decoded text or null. */
    public static final class Expected {
        private final int[] ids;
        private final String decoded;

        Expected(int[] ids, String decoded) {
            this.ids = ids;
            this.decoded = decoded;
        }

        public int[] ids() {
            return ids.clone();
        }

        public String decoded() {
            return decoded;
        }
    }

    /** The text for a test name: at most 40 chars, line breaks and tabs escaped, quoted. */
    public static String printable(String text) {
        String shown = text.length() > 40 ? text.substring(0, 40) + "..." : text;
        return "\"" + shown.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t") + "\"";
    }

    /**
     * Fails unless the counts and the byte decoding agree with {@code ids}: countTokens of the text
     * is their number, decode is decodeBytes as UTF-8, and countBytes is decodeBytes' length.
     */
    public static void checkCountsAndBytes(
            String text,
            int[] ids,
            ToIntFunction<String> countTokens,
            Function<int[], String> decode,
            Function<int[], byte[]> decodeBytes,
            ToIntFunction<int[]> countBytes) {
        int count = countTokens.applyAsInt(text);
        if (count != ids.length) {
            throw new AssertionError("countTokens: expected " + ids.length + " but was " + count);
        }
        byte[] bytes = decodeBytes.apply(ids);
        String decoded = decode.apply(ids);
        String fromBytes = new String(bytes, StandardCharsets.UTF_8);
        if (!Objects.equals(decoded, fromBytes)) {
            throw new AssertionError(
                    "decode agrees with decodeBytes: "
                            + printable(decoded)
                            + " vs "
                            + printable(fromBytes));
        }
        int byteCount = countBytes.applyAsInt(ids);
        if (byteCount != bytes.length) {
            throw new AssertionError(
                    "countBytes: expected " + bytes.length + " but was " + byteCount);
        }
    }
}
