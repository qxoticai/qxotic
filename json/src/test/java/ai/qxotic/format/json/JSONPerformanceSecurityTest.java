package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Performance and security tests for JSON parser. Tests memory usage, recursion depth, and
 * malicious input handling.
 */
class JSONPerformanceSecurityTest {

    @Test
    void testLargeInputMemorySafety() {
        // Very long string (100KB) should parse without OOM
        StringBuilder sb = new StringBuilder();
        sb.append("\"");
        for (int i = 0; i < 100000; i++) {
            sb.append("x");
        }
        sb.append("\"");

        Object parsed = JSON.parse(sb.toString());
        assertTrue(parsed instanceof String);
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

        parsed = JSON.parse(sb.toString());
        assertTrue(parsed instanceof List);
        List<?> list = (List<?>) parsed;
        assertEquals(10000, list.size());
    }

    @Test
    void testDeepRecursionSafety() {
        // Test near maximum depth (default 1000)
        // Use smaller depth to avoid Java recursion limits
        int depth = 200;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < depth; i++) {
            sb.append("[");
        }
        sb.append("null");
        for (int i = 0; i < depth; i++) {
            sb.append("]");
        }

        // Should parse successfully at default max depth (1000)
        Object result = JSON.parse(sb.toString());
        assertNotNull(result);

        // Should fail if we exceed max depth
        // Use 1200 to exceed default 1000 but not cause Java stack overflow
        int exceededDepth = 1200;
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < exceededDepth; i++) {
            sb2.append("[");
        }
        sb2.append("null");
        for (int i = 0; i < exceededDepth; i++) {
            sb2.append("]");
        }

        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse(sb2.toString()));
        assertNotNull(e.getMessage());

        // Should work with increased max depth
        JSON.ParseOptions options = JSON.ParseOptions.create().maxParsingDepth(2000);
        result = JSON.parse(sb2.toString(), options);
        assertNotNull(result);
    }

    @Test
    void testNestedObjectsDepth() {
        // Deep nested objects
        int depth = 500;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < depth; i++) {
            sb.append("{\"a\":");
        }
        sb.append("null");
        for (int i = 0; i < depth; i++) {
            sb.append("}");
        }

        Object result = JSON.parse(sb.toString());
        assertNotNull(result);
    }

    @Test
    void testMaliciousLargeExponent() {
        // Very large exponents should be handled safely
        Object parsed = JSON.parse("1e999999");
        assertTrue(parsed instanceof BigDecimal);

        parsed = JSON.parse("1e-999999");
        assertTrue(parsed instanceof BigDecimal);

        // Test overflow behavior
        try {
            parsed =
                    JSON.parse(
                            "999999999999999999999999999999999999999999999999999999999999999999999999999999999999");
            // Should be BigInteger
            assertTrue(parsed instanceof BigInteger);
        } catch (JSON.ParseException e) {
            // Also acceptable to reject
            assertNotNull(e.getMessage());
        }
    }

    @Test
    void testResourceExhaustionPrevention() {
        // Test that we don't hang on extremely large inputs
        // This is more of a smoke test - actual performance testing would require timing

        // Create pathological input with many escapes
        StringBuilder sb = new StringBuilder();
        sb.append("\"");
        for (int i = 0; i < 10000; i++) {
            sb.append("\\u");
        }
        sb.append("\"");

        // Should throw ParseException quickly, not hang
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse(sb.toString()));
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

            Object parsed = JSON.parse(sb.toString());
            assertTrue(parsed instanceof List);
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
            Object parsed = JSON.parse(beyondRange);
            assertTrue(parsed instanceof String);
        } catch (JSON.ParseException e) {
            assertNotNull(e.getMessage());
        }

        // Test very high Unicode (beyond BMP)
        String highUnicode = "\"\uD800\uDC00\""; // Valid surrogate pair for U+10000
        Object parsed = JSON.parse(highUnicode);
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

        String json = JSON.stringify(largeMap);
        assertNotNull(json);
        assertTrue(json.length() > 1000);

        // Round-trip should work
        Object parsed = JSON.parse(json);
        assertTrue(parsed instanceof Map);
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
                                        Object parsed = JSON.parse(testJson);
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
                Object parsed = JSON.parse(number);
                // Should parse as BigDecimal or BigInteger
                assertTrue(parsed instanceof BigDecimal || parsed instanceof BigInteger);
            } catch (JSON.ParseException e) {
                // Also acceptable - implementation may reject extreme values
                assertNotNull(e.getMessage());
            }
        }
    }

    @Test
    void testEmptyAndWhitespaceOnlyInput() {
        // Empty input should throw
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(""));
        assertNotNull(e.getMessage());

        // Whitespace-only input should throw
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("   "));
        assertNotNull(e.getMessage());

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("\n\t\r "));
        assertNotNull(e.getMessage());
    }

    @Test
    void testParseOptionsMemorySafety() {
        // Test that ParseOptions doesn't cause memory issues
        JSON.ParseOptions options =
                JSON.ParseOptions.create().useBigDecimalForFloats().maxParsingDepth(5000);

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

        Object parsed = JSON.parse(sb.toString(), options);
        assertNotNull(parsed);
    }
}

class AtomicInteger {
    private int value = 0;

    public synchronized void incrementAndGet() {
        value++;
    }

    public synchronized int get() {
        return value;
    }
}
