// Thin JSON facade over com.qxotic:json: parse/stringify with this project's null and
// number conventions (Json.NULL <-> Java null, decimals as double). The single place the
// external JSON library is adapted to the engine's Map/List/null value model.
package com.qxotic.jinfer.server;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.*;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class JsonCodec {
    private static final Json.ParseOptions OPTIONS =
            Json.ParseOptions.defaults().decimalsAsBigDecimal(false);

    static Object parse(String text) {
        return fromLibrary(Json.parse(text, OPTIONS));
    }

    static String stringify(Object value) {
        return Json.stringify(toLibrary(value));
    }

    /** Json.NULL -> Java null, in place (the parser's containers are mutable). */
    private static Object fromLibrary(Object value) {
        if (value == Json.NULL) {
            return null;
        }
        if (value instanceof Map<?, ?> map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> object = (Map<String, Object>) map;
            for (Map.Entry<String, Object> entry : object.entrySet()) {
                entry.setValue(fromLibrary(entry.getValue()));
            }
        } else if (value instanceof List<?> list) {
            @SuppressWarnings("unchecked")
            List<Object> array = (List<Object>) list;
            array.replaceAll(JsonCodec::fromLibrary);
        }
        return value;
    }

    /** Java null -> Json.NULL, plus the lenient coercions the previous printer had. */
    private static Object toLibrary(Object value) {
        if (value == null) {
            return Json.NULL;
        }
        if (value instanceof String || value instanceof Number || value instanceof Boolean) {
            return value;
        }
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> object = LinkedHashMap.newLinkedHashMap(map.size());
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                object.put(String.valueOf(entry.getKey()), toLibrary(entry.getValue()));
            }
            return object;
        }
        if (value instanceof Iterable<?> iterable) {
            List<Object> array = new ArrayList<>();
            for (Object item : iterable) {
                array.add(toLibrary(item));
            }
            return array;
        }
        return String.valueOf(value);
    }
}
