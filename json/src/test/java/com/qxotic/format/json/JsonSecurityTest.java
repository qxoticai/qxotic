package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Resource limits, malicious input and fuzzing. */
class JsonSecurityTest {

    @Test
    void testLargeInputMemorySafety() {
        // Very long string (100KB) should parse without OOM
        StringBuilder sb = new StringBuilder();
        sb.append("\"");
        sb.append("x".repeat(100000));
        sb.append("\"");

        Object parsed = Json.parse(sb.toString());
        assertInstanceOf(String.class, parsed);
        String result = (String) parsed;
        assertEquals(100000, result.length());

        // Large number of elements in array
        sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < 10000; i++) {
            if (i > 0) sb.append(",");
            sb.append(i);
        }
        sb.append("]");

        parsed = Json.parse(sb.toString());
        assertInstanceOf(List.class, parsed);
        List<?> list = (List<?>) parsed;
        assertEquals(10000, list.size());
    }

    @Test
    void testDeepRecursionSafety() {
        // Test near maximum depth (default 1000)
        // Use smaller depth to avoid Java recursion limits
        int depth = 200;
        String sb = "[".repeat(depth) + "null" + "]".repeat(depth);
        // Should parse successfully at default max depth (1000)
        Object result = Json.parse(sb);
        assertNotNull(result);

        // Should fail if we exceed max depth
        // Use 1200 to exceed default 1000 but not cause Java stack overflow
        int exceededDepth = 1200;
        String sb2 = "[".repeat(exceededDepth) + "null" + "]".repeat(exceededDepth);
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(sb2));
        assertNotNull(e.getMessage());

        // Should work with increased max depth
        Json.ParseOptions options = Json.ParseOptions.defaults().maxDepth(2000);
        result = Json.parse(sb2, options);
        assertNotNull(result);
    }

    @Test
    void testNestedObjectsDepth() {
        // Deep nested objects
        int depth = 500;
        String sb = "{\"a\":".repeat(depth) + "null" + "}".repeat(depth);
        Object result = Json.parse(sb);
        assertNotNull(result);
    }

    @Test
    void testMaliciousLargeExponent() {
        // Very large exponents should be handled safely
        Object parsed = Json.parse("1e999999");
        assertInstanceOf(BigDecimal.class, parsed);

        parsed = Json.parse("1e-999999");
        assertInstanceOf(BigDecimal.class, parsed);

        // Test overflow behavior
        try {
            parsed =
                    Json.parse(
                            "999999999999999999999999999999999999999999999999999999999999999999999999999999999999");
            // Should be BigInteger
            assertInstanceOf(BigInteger.class, parsed);
        } catch (Json.ParseException e) {
            // Also acceptable to reject
            assertNotNull(e.getMessage());
        }
    }

    @Test
    void testResourceExhaustionPrevention() {
        // Test that we don't hang on extremely large inputs
        // This is more of a smoke test - actual performance testing would require timing

        // Create pathological input with many escapes
        String sb = "\"" + "\\u".repeat(10000) + "\"";

        // Should throw ParseException quickly, not hang
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(sb));
        assertNotNull(e.getMessage());
    }

    @Test
    void testMemoryLeakResistance() {
        // Parse multiple large inputs to check for memory leaks
        // (This is a smoke test - real memory leak testing would require heap analysis)

        for (int iteration = 0; iteration < 100; iteration++) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            for (int i = 0; i < 1000; i++) {
                if (i > 0) sb.append(",");
                sb.append("{\"id\":").append(i).append(",\"value\":\"test").append(i).append("\"}");
            }
            sb.append("]");

            Object parsed = Json.parse(sb.toString());
            assertInstanceOf(List.class, parsed);
            List<?> list = (List<?>) parsed;
            assertEquals(1000, list.size());
        }
    }

    @Test
    void testInvalidUnicodeBoundaryHandling() {
        // Test code points beyond valid Unicode range
        String beyondRange = "\"\u00110000\"";
        // This should either parse (if implementation accepts) or throw
        try {
            Object parsed = Json.parse(beyondRange);
            assertInstanceOf(String.class, parsed);
        } catch (Json.ParseException e) {
            assertNotNull(e.getMessage());
        }

        // Test very high Unicode (beyond BMP)
        String highUnicode = "\"\uD800\uDC00\""; // Valid surrogate pair for U+10000
        Object parsed = Json.parse(highUnicode);
        assertEquals("\uD800\uDC00", parsed);
    }

    @Test
    void testStringifyLargeStructures() {
        // Test stringify with large/complex structures
        Map<String, Object> largeMap = new HashMap<>();
        List<Object> largeList = new ArrayList<>();

        for (int i = 0; i < 1000; i++) {
            Map<String, Object> item = new HashMap<>();
            item.put("id", (long) i);
            item.put("name", "Item " + i);
            item.put("value", i * 1.5);
            largeList.add(item);
        }

        largeMap.put("data", largeList);
        largeMap.put("count", 1000L);
        largeMap.put("timestamp", new BigDecimal("1234567890.123456"));

        String json = Json.stringify(largeMap);
        assertNotNull(json);
        assertTrue(json.length() > 1000);

        // Round-trip should work
        Object parsed = Json.parse(json);
        assertInstanceOf(Map.class, parsed);
    }

    @Test
    void testConcurrentParsingSafety() throws InterruptedException {
        // Test that parser can handle concurrent usage
        // (This is a basic test - real concurrency testing would be more complex)

        final String testJson = "[1,2,3,{\"a\":\"b\"}]";
        final int threadCount = 10;
        final int iterations = 100;

        List<Thread> threads = new ArrayList<>();
        final AtomicInteger successCount = new AtomicInteger();
        final AtomicInteger errorCount = new AtomicInteger();

        for (int t = 0; t < threadCount; t++) {
            Thread thread =
                    new Thread(
                            () -> {
                                for (int i = 0; i < iterations; i++) {
                                    try {
                                        Object parsed = Json.parse(testJson);
                                        assertNotNull(parsed);
                                        successCount.incrementAndGet();
                                    } catch (Exception e) {
                                        errorCount.incrementAndGet();
                                    }
                                }
                            });
            threads.add(thread);
        }

        for (Thread thread : threads) {
            thread.start();
        }

        for (Thread thread : threads) {
            thread.join();
        }

        assertEquals(threadCount * iterations, successCount.get() + errorCount.get());
        assertEquals(threadCount * iterations, successCount.get()); // All should succeed
    }

    @Test
    void testInvalidNumberCrashPrevention() {
        // Numbers that might cause overflow or other issues
        String[] problematicNumbers = {
            "1e999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999",
            "-1e999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999",
            "999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999.999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999",
        };

        for (String number : problematicNumbers) {
            try {
                Object parsed = Json.parse(number);
                // Should parse as BigDecimal or BigInteger
                assertTrue(parsed instanceof BigDecimal || parsed instanceof BigInteger);
            } catch (Json.ParseException e) {
                // Also acceptable - implementation may reject extreme values
                assertNotNull(e.getMessage());
            }
        }
    }

    @Test
    void testEmptyAndWhitespaceOnlyInput() {
        // Empty input should throw
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(""));
        assertNotNull(e.getMessage());

        // Whitespace-only input should throw
        e = assertThrows(Json.ParseException.class, () -> Json.parse("   "));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\n\t\r "));
        assertNotNull(e.getMessage());
    }

    @Test
    void testParseOptionsMemorySafety() {
        // Test that ParseOptions doesn't cause memory issues
        Json.ParseOptions options =
                Json.ParseOptions.defaults().decimalsAsBigDecimal(true).maxDepth(5000);

        // Large structure with increased depth
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < 1000; i++) {
            if (i > 0) sb.append(",");
            sb.append("[");
            for (int j = 0; j < 10; j++) {
                if (j > 0) sb.append(",");
                sb.append("{\"layer\":").append(j).append("}");
            }
            sb.append("]");
        }
        sb.append("]");

        Object parsed = Json.parse(sb.toString(), options);
        assertNotNull(parsed);
    }

    private static final long SEED = 0xC0FFEE1234ABL;

    @Test
    void fuzzValidRoundTrip() {
        assertTimeoutPreemptively(
                Duration.ofSeconds(60), // a hang guard, not a budget: cold JIT on a shared runner
                () -> {
                    Random rnd = new Random(SEED);
                    for (int i = 0; i < 100_000; i++) {
                        Object value = randomValue(rnd, 0, 4);

                        String json = Json.stringify(value, rnd.nextBoolean());
                        assertTrue(Json.isValid(json), "Generated JSON should be valid");

                        Object parsed = Json.parse(json);
                        String canonical = Json.stringify(parsed, false);

                        Object parsedAgain = Json.parse(canonical);
                        String canonicalAgain = Json.stringify(parsedAgain, false);

                        assertEquals(canonical, canonicalAgain, "Canonical output must be stable");
                    }
                });
    }

    @Test
    void fuzzMutatedInputsNeverCrash() {
        assertTimeoutPreemptively(
                Duration.ofSeconds(60), // a hang guard, not a budget: cold JIT on a shared runner
                () -> {
                    Random rnd = new Random(SEED ^ 0x5EEDBEEFL);
                    for (int i = 0; i < 200_000; i++) {
                        String base = Json.stringify(randomValue(rnd, 0, 3), false);
                        String mutated = mutate(base, rnd);

                        try {
                            Json.parse(mutated);
                        } catch (RuntimeException e) {
                            if (!(e instanceof Json.ParseException)) {
                                fail("Unexpected exception type: " + e.getClass().getName());
                            }
                        }
                    }
                });
    }

    private static Object randomValue(Random rnd, int depth, int maxDepth) {
        if (depth >= maxDepth) {
            return randomScalar(rnd);
        }

        int pick = rnd.nextInt(10);
        if (pick < 6) {
            return randomScalar(rnd);
        }
        if (pick < 8) {
            int size = rnd.nextInt(5);
            List<Object> list = new ArrayList<>(size);
            for (int i = 0; i < size; i++) {
                list.add(randomValue(rnd, depth + 1, maxDepth));
            }
            return list;
        }

        int size = rnd.nextInt(5);
        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < size; i++) {
            map.put(randomKey(rnd, i), randomValue(rnd, depth + 1, maxDepth));
        }
        return map;
    }

    private static Object randomScalar(Random rnd) {
        switch (rnd.nextInt(10)) {
            case 0:
                return Json.NULL;
            case 1:
                return rnd.nextBoolean();
            case 2:
                return rnd.nextLong();
            case 3:
                return BigInteger.valueOf(rnd.nextLong())
                        .multiply(BigInteger.valueOf(rnd.nextInt(10) + 1));
            case 4:
                return randomBigDecimal(rnd);
            case 5:
                return randomFiniteDouble(rnd);
            case 6:
                return (float) randomFiniteDouble(rnd);
            default:
                return randomString(rnd);
        }
    }

    private static BigDecimal randomBigDecimal(Random rnd) {
        long whole = Math.abs(rnd.nextLong() % 1_000_000L);
        long frac = Math.abs(rnd.nextLong() % 1_000_000L);
        String s = whole + "." + frac;
        if (rnd.nextBoolean()) {
            s = "-" + s;
        }
        return new BigDecimal(s);
    }

    private static double randomFiniteDouble(Random rnd) {
        double d;
        do {
            d = rnd.nextGaussian() * 1_000_000;
        } while (Double.isNaN(d) || Double.isInfinite(d));
        return d;
    }

    private static String randomKey(Random rnd, int index) {
        return "k" + index + "_" + Integer.toHexString(rnd.nextInt());
    }

    private static String randomString(Random rnd) {
        int len = rnd.nextInt(16);
        StringBuilder sb = new StringBuilder(len);
        for (int i = 0; i < len; i++) {
            int pick = rnd.nextInt(8);
            switch (pick) {
                case 0:
                    sb.append((char) ('a' + rnd.nextInt(26)));
                    break;
                case 1:
                    sb.append((char) ('A' + rnd.nextInt(26)));
                    break;
                case 2:
                    sb.append((char) ('0' + rnd.nextInt(10)));
                    break;
                case 3:
                    sb.append('"');
                    break;
                case 4:
                    sb.append('\\');
                    break;
                case 5:
                    sb.append('\n');
                    break;
                case 6:
                    sb.append('\t');
                    break;
                default:
                    sb.append((char) (0x00A1 + rnd.nextInt(0x00FF - 0x00A1)));
                    break;
            }
        }
        return sb.toString();
    }

    private static String mutate(String input, Random rnd) {
        StringBuilder sb = new StringBuilder(input);
        int operations = 1 + rnd.nextInt(3);

        for (int i = 0; i < operations; i++) {
            int op = rnd.nextInt(3);
            switch (op) {
                case 0:
                    // insert
                    int insertAt = rnd.nextInt(sb.length() + 1);
                    sb.insert(insertAt, randomMutationChar(rnd));
                    break;
                case 1:
                    // delete
                    if (sb.length() > 0) {
                        int deleteAt = rnd.nextInt(sb.length());
                        sb.deleteCharAt(deleteAt);
                    }
                    break;
                default:
                    // replace
                    if (sb.length() > 0) {
                        int replaceAt = rnd.nextInt(sb.length());
                        sb.setCharAt(replaceAt, randomMutationChar(rnd));
                    }
                    break;
            }
        }

        return sb.toString();
    }

    private static char randomMutationChar(Random rnd) {
        char[] chars = {'{', '}', '[', ']', '"', ':', ',', '\\', '\n', 'x', '0', '-', '+'};
        return chars[rnd.nextInt(chars.length)];
    }
}
