package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** The value model {@link JsonCodec#parse} promises - the one a caller can hold on to. */
class JsonCodecTest {

    @Test
    void parseIsTheFrozenValueModel() {
        Object parsed = JsonCodec.parse("{\"z\": 1, \"a\": 2.5, \"n\": null, \"l\": [1, \"x\"]}");
        Map<?, ?> object = assertInstanceOf(Map.class, parsed);
        assertEquals(List.of("z", "a", "n", "l"), List.copyOf(object.keySet()), "document order");
        assertInstanceOf(Long.class, object.get("z"));
        assertInstanceOf(Double.class, object.get("a"), "every decimal is a Double");
        assertTrue(object.containsKey("n"), "JSON null is a present key");
        assertNull(object.get("n"), "holding Java null");
        List<?> list = assertInstanceOf(List.class, object.get("l"));
        assertEquals(1L, list.get(0));
        // deeply unmodifiable: the same frozen form the records hold
        assertThrows(
                UnsupportedOperationException.class,
                () -> ((Map<String, Object>) object).put("k", 1));
        assertThrows(UnsupportedOperationException.class, () -> ((List<Object>) list).add(1));
        assertEquals("{\"z\":1,\"a\":2.5,\"n\":null,\"l\":[1,\"x\"]}", JsonCodec.stringify(parsed));
    }

    @ParameterizedTest
    @CsvSource({
        "1, java.lang.Long",
        "-9223372036854775808, java.lang.Long",
        "9223372036854775807, java.lang.Long",
        "-9223372036854775809, java.math.BigInteger",
        "9223372036854775808, java.math.BigInteger"
    })
    void integersPreserveTheirTypeAndValue(String json, Class<? extends Number> expectedType) {
        Number parsed = assertInstanceOf(expectedType, JsonCodec.parse(json));
        assertEquals(json, parsed.toString());
        assertEquals(json, JsonCodec.stringify(parsed));
    }
}
