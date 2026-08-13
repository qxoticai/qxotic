package com.qxotic.jinfer.x.chat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Defensive copies for the JSON-shaped values crossing the chat API. */
final class JsonValues {
    private JsonValues() {}

    static Map<String, Object> object(Map<String, ?> source) {
        Objects.requireNonNull(source, "source");
        LinkedHashMap<String, Object> copy = LinkedHashMap.newLinkedHashMap(source.size());
        source.forEach((key, value) -> copy.put(key, freeze(value)));
        return Collections.unmodifiableMap(copy);
    }

    private static Object freeze(Object value) {
        if (value == null
                || value instanceof String
                || value instanceof Number
                || value instanceof Boolean) return value;
        if (value instanceof Map<?, ?> map) {
            LinkedHashMap<String, Object> copy = LinkedHashMap.newLinkedHashMap(map.size());
            map.forEach(
                    (key, item) -> {
                        if (!(key instanceof String text))
                            throw new IllegalArgumentException(
                                    "JSON object key is not a string: " + key);
                        copy.put(text, freeze(item));
                    });
            return Collections.unmodifiableMap(copy);
        }
        if (value instanceof List<?> list) {
            ArrayList<Object> copy = new ArrayList<>(list.size());
            list.forEach(item -> copy.add(freeze(item)));
            return Collections.unmodifiableList(copy);
        }
        throw new IllegalArgumentException("not a JSON value: " + value.getClass().getName());
    }
}
