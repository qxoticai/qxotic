package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** {@link Tool#parameters()}: the schema source for every schema-bound forced-call grammar. */
final class ToolParametersTest {

    @Test
    void readsParametersFromABareFunctionObject() {
        Tool t = new Tool("f", "{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"x\":1}}");
        assertEquals(Map.of("type", "object", "x", 1L), t.parameters());
    }

    @Test
    void unwrapsTheTypeFunctionEnvelope() {
        Tool t =
                new Tool(
                        "f",
                        "{\"type\":\"function\",\"function\":{\"name\":\"f\","
                                + "\"parameters\":{\"type\":\"object\"}}}");
        assertEquals(Map.of("type", "object"), t.parameters());
    }

    @Test
    void aToolDeclaringNoParametersGetsTheEmptyObjectSchema() {
        // the guarantee behind "a forced no-parameter call cannot be decorated": the grammar
        // compiled from this admits exactly {}
        assertEquals(Map.of("type", "object"), new Tool("f", "{\"name\":\"f\"}").parameters());
        assertEquals(
                Map.of("type", "object"),
                new Tool("f", "{\"name\":\"f\",\"parameters\":\"junk\"}").parameters());
    }
}
