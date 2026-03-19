package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

class JsonFuzzerTest {

    private static final long SEED = 0xC0FFEE1234ABL;

    @Test
    void fuzzValidRoundTrip() {
        assertTimeoutPreemptively(
                Duration.ofSeconds(5),
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
                Duration.ofSeconds(5),
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
