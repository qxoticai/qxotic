package com.qxotic.format.json;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Documentation snippets for JSON module.
 *
 * <p>All code in docs/ must reference snippets from this file. Never write code directly in
 * Markdown.
 */
@SuppressWarnings("unused")
class Snippets {

    // ========== QUICK START ==========

    void quickStart() {
        // --8<-- [start:quick-start]
        // Parse JSON string
        Map<String, Object> config = JSON.parseMap("{\"host\":\"localhost\",\"port\":8080}");

        String host = (String) config.get("host");
        Long port = (Long) config.get("port");

        // Serialize back to JSON
        String json = JSON.stringify(config);
        // --8<-- [end:quick-start]
    }

    // ========== PARSING ==========

    void parseGeneric() {
        // --8<-- [start:parse-generic]
        Object value = JSON.parse("{\"name\":\"alice\",\"age\":30}");

        if (value instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> obj = (Map<String, Object>) value;
            System.out.println(obj.get("name")); // "alice"
        }
        // --8<-- [end:parse-generic]
    }

    void parseTypedRoot() {
        // --8<-- [start:parse-typed-root]
        // Require specific root type - throws if mismatch
        Map<String, Object> obj = JSON.parseMap("{\"x\":1,\"y\":2}");
        List<Object> arr = JSON.parseList("[1, 2, 3]");
        String str = JSON.parseString("\"hello world\"");
        Number num = JSON.parseNumber("3.14159");
        boolean bool = JSON.parseBoolean("true");

        // Generic typed parse
        Map<?, ?> alsoObj = JSON.parseMap("{\"status\":\"ok\"}");
        // --8<-- [end:parse-typed-root]
    }

    void parseObjectAccess() {
        // --8<-- [start:parse-object-access]
        String json =
                "{"
                        + "\"model\": \"llama-3\","
                        + "\"config\": {\"hidden_size\": 4096, \"layers\": 32},"
                        + "\"vocab\": [32000, 32001, 32002]"
                        + "}";

        Map<String, Object> root = JSON.parseMap(json);

        String model = (String) root.get("model");

        @SuppressWarnings("unchecked")
        Map<String, Object> config = (Map<String, Object>) root.get("config");
        Long hiddenSize = (Long) config.get("hidden_size");

        @SuppressWarnings("unchecked")
        List<Object> vocab = (List<Object>) root.get("vocab");
        // --8<-- [end:parse-object-access]
    }

    void parseValidation() {
        // --8<-- [start:parse-validation]
        // Check validity without throwing
        boolean valid = JSON.isValid("{\"x\":1}"); // true
        boolean invalid = JSON.isValid("{\"x\":}"); // false

        // Use in guards
        String userInput = getUserInput();
        if (JSON.isValid(userInput)) {
            Map<String, Object> data = JSON.parseMap(userInput);
            process(data);
        }
        // --8<-- [end:parse-validation]
    }

    private String getUserInput() {
        return "{}";
    }

    private void process(Map<String, Object> data) {}

    // ========== PARSE OPTIONS ==========

    void parseOptionsBasic() {
        // --8<-- [start:parse-options-basic]
        // Default options: BigDecimal for decimals, max depth 1000
        JSON.ParseOptions defaults = JSON.ParseOptions.defaults();

        // Customize options
        JSON.ParseOptions options =
                JSON.ParseOptions.defaults()
                        .decimalsAsBigDecimal(false) // Use Double instead
                        .maxDepth(100) // Limit nesting
                        .failOnDuplicateKeys(true); // Reject duplicate keys

        Object value = JSON.parse("[1, 2, 3]", options);
        // --8<-- [end:parse-options-basic]
    }

    void parseOptionsDecimals() {
        // --8<-- [start:parse-options-decimals]
        // Default: precise decimal handling
        Number precise = JSON.parseNumber("0.1");
        // precise is BigDecimal

        // Alternative: use Double (faster, potential precision loss)
        JSON.ParseOptions doubleMode = JSON.ParseOptions.defaults().decimalsAsBigDecimal(false);
        Number approx = JSON.parseNumber("0.1", doubleMode);
        // approx is Double
        // --8<-- [end:parse-options-decimals]
    }

    void parseOptionsDepth() {
        // --8<-- [start:parse-options-depth]
        // Deeply nested JSON
        String nested = "[[[[[[[[[[1]]]]]]]]]]";

        // Limit depth to prevent stack overflow on malicious input
        JSON.ParseOptions shallow = JSON.ParseOptions.defaults().maxDepth(10);

        try {
            JSON.parse(nested, shallow);
        } catch (JSON.ParseException e) {
            // "Maximum parsing depth exceeded"
        }
        // --8<-- [end:parse-options-depth]
    }

    void parseOptionsDuplicateKeys() {
        // --8<-- [start:parse-options-duplicate]
        String json = "{\"a\":1,\"a\":2}";

        // Default: last key wins
        Map<String, Object> defaultBehavior = JSON.parseMap(json);
        // defaultBehavior.get("a") == 2

        // Strict mode: reject duplicates
        JSON.ParseOptions strict = JSON.ParseOptions.defaults().failOnDuplicateKeys(true);
        try {
            JSON.parse(json, strict);
        } catch (JSON.ParseException e) {
            // "Duplicate key: 'a'"
        }
        // --8<-- [end:parse-options-duplicate]
    }

    // ========== DATA MODEL ==========

    void dataModel() {
        // --8<-- [start:data-model]
        Map<String, Object> obj = JSON.parseMap("{\"int\":42,\"decimal\":3.14,\"big\":9999999999}");

        // Integer fits in long
        Long intVal = (Long) obj.get("int");

        // Decimal is BigDecimal by default
        BigDecimal decimalVal = (BigDecimal) obj.get("decimal");

        // Large integer becomes BigInteger
        BigInteger bigVal = (BigInteger) obj.get("big");

        // Null handling
        String jsonWithNull = "{\"value\":null}";
        Map<String, Object> withNull = JSON.parseMap(jsonWithNull);
        Object n = withNull.get("value");
        // n == JSON.NULL (not Java null!)
        // --8<-- [end:data-model]
    }

    void nullHandling() {
        // --8<-- [start:null-handling]
        Map<String, Object> obj = JSON.parseMap("{\"present\":null}");
        obj.put("absent", null); // Simulating missing key

        Object present = obj.get("present");
        if (JSON.isNull(present)) {
            // JSON null value (key exists, value is null)
        }

        Object absent = obj.get("absent");
        if (absent == null) {
            // Key doesn't exist or was set to Java null
        }

        // Serialize JSON.NULL back to "null"
        String back = JSON.stringify(Map.of("value", JSON.NULL));
        // back == "{\"value\":null}"
        // --8<-- [end:null-handling]
    }

    // ========== SERIALIZATION ==========

    void stringifyBasic() {
        // --8<-- [start:stringify-basic]
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("name", "alice");
        data.put("active", true);
        data.put("count", 42);
        data.put("tags", List.of("a", "b", "c"));

        // Compact output
        String compact = JSON.stringify(data);
        // {"name":"alice","active":true,"count":42,"tags":["a","b","c"]}

        // Pretty-printed
        String pretty = JSON.stringify(data, true);
        // --8<-- [end:stringify-basic]
    }

    void stringifyPretty() {
        // --8<-- [start:stringify-pretty]
        Map<String, Object> config =
                Map.of(
                        "model", Map.of("name", "llama", "size", 7),
                        "training", Map.of("epochs", 100, "lr", 0.001));

        String pretty = JSON.stringify(config, true);
        // {
        //   "model" : {
        //     "name" : "llama",
        //     "size" : 7
        //   },
        //   "training" : {
        //     "epochs" : 100,
        //     "lr" : 0.001
        //   }
        // }
        // --8<-- [end:stringify-pretty]
    }

    void roundTrip() {
        // --8<-- [start:round-trip]
        // Parse
        Map<String, Object> original = JSON.parseMap("{\"x\":1,\"y\":2}");

        // Modify
        original.put("z", 3);

        // Serialize
        String json = JSON.stringify(original);

        // Parse again
        Map<String, Object> parsed = JSON.parseMap(json);
        // --8<-- [end:round-trip]
    }

    // ========== ERROR HANDLING ==========

    void errorHandling() {
        // --8<-- [start:error-handling]
        try {
            JSON.parse("{\"broken\":,}");
        } catch (JSON.ParseException e) {
            // Error message with location
            String message = e.getMessage();
            // "Line 1, Column 10: Expected value"
            // "{\"broken\":,}"
            //           ^

            // Position details
            int position = e.getPosition(); // 10
            int line = e.getLine(); // 1
            int column = e.getColumn(); // 10
        }
        // --8<-- [end:error-handling]
    }

    void errorTypes() {
        // --8<-- [start:error-types]
        // Syntax errors
        assertThrows(() -> JSON.parse("{")); // Unexpected end of input
        assertThrows(() -> JSON.parse("[1,]")); // Expected value
        assertThrows(() -> JSON.parse("[1 2]")); // Expected ',' or ']'
        assertThrows(() -> JSON.parse("{\"a\"}")); // Expected ':'

        // Type mismatches
        assertThrows(() -> JSON.parseMap("[]")); // Expected JSON object at root
        assertThrows(() -> JSON.parseList("{}")); // Expected JSON array at root
        // --8<-- [end:error-types]
    }

    private void assertThrows(Runnable r) {
        try {
            r.run();
            throw new AssertionError("Expected exception");
        } catch (Exception expected) {
        }
    }

    // ========== VALIDATION ==========

    void stringifyValidation() {
        // --8<-- [start:stringify-validation]
        // Cyclic structures are rejected
        Map<String, Object> cycle = new LinkedHashMap<>();
        cycle.put("self", cycle);

        try {
            JSON.stringify(cycle);
        } catch (IllegalArgumentException e) {
            // "Cannot serialize cyclic structure"
        }

        // NaN and Infinity are rejected
        try {
            JSON.stringify(Map.of("value", Double.NaN));
        } catch (IllegalArgumentException e) {
            // "Cannot serialize NaN/Infinity"
        }
        // --8<-- [end:stringify-validation]
    }

    // ========== GGUF INTEGRATION ==========

    void ggufIntegration() {
        // --8<-- [start:gguf-integration]
        // Typical pattern: read GGUF metadata JSON
        String metadataJson =
                "{"
                        + "\"general.name\": \"Llama-3-8B\","
                        + "\"general.architecture\": \"llama\","
                        + "\"llama.context_length\": 8192,"
                        + "\"llama.embedding_length\": 4096"
                        + "}";

        Map<String, Object> meta = JSON.parseMap(metadataJson);

        String name = (String) meta.get("general.name");
        Long contextLength = (Long) meta.get("llama.context_length");
        // --8<-- [end:gguf-integration]
    }

    // ========== UNICODE ==========

    void unicodeHandling() {
        // --8<-- [start:unicode]
        // Unicode in strings
        Map<String, Object> obj =
                JSON.parseMap("{\"emoji\":\"" + "\uD83C\uDF0D" + "\",\"jp\":\"こんにちは\"}");

        // Round-trips correctly
        String encoded = JSON.stringify(obj);
        Map<String, Object> decoded = JSON.parseMap(encoded);

        // Escape sequences
        String withEscape = JSON.parseString("\"Hello\\nWorld\""); // "Hello\nWorld"
        // --8<-- [end:unicode]
    }
}
