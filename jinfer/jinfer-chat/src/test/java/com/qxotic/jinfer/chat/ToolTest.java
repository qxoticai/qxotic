package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class ToolTest {

    @Test
    void readsBareAndWrappedParameterSchemas() {
        Map<String, Object> schema = Map.of("type", "object", "required", List.of("city"));
        assertEquals(schema, new Tool("weather", Map.of("parameters", schema)).parameters());
        assertEquals(
                schema,
                new Tool(
                                "weather",
                                Map.of(
                                        "type",
                                        "function",
                                        "function",
                                        Map.of("name", "weather", "parameters", schema)))
                        .parameters());
    }

    @Test
    void missingOrInvalidParametersMeanAnEmptyObject() {
        assertEquals(Map.of("type", "object"), new Tool("f", Map.of()).parameters());
        assertEquals(
                Map.of("type", "object"),
                new Tool("f", Map.of("parameters", "invalid")).parameters());
        assertThrows(NullPointerException.class, () -> new Tool("f", null));
    }

    @Test
    @SuppressWarnings("unchecked")
    void definitionsAreDeeplyImmutableSnapshots() {
        List<Object> required = new ArrayList<>(List.of("city"));
        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("required", required);
        Tool tool = new Tool("weather", Map.of("parameters", parameters));
        required.add("unit");
        parameters.put("extra", true);

        assertEquals(List.of("city"), tool.parameters().get("required"));
        assertThrows(
                UnsupportedOperationException.class, () -> tool.definition().put("extra", true));
        assertThrows(
                UnsupportedOperationException.class,
                () -> ((List<Object>) tool.parameters().get("required")).add("unit"));
    }

    @Test
    void factoryBuildsAFunctionDefinition() {
        Map<String, Object> parameters =
                Map.of(
                        "type",
                        "object",
                        "properties",
                        Map.of("city", Map.of("type", "string")),
                        "required",
                        List.of("city"));

        Tool tool = Tool.of("get_weather", "Current weather", parameters);

        assertEquals("get_weather", tool.name());
        assertEquals(parameters, tool.parameters());
        assertEquals(
                Map.of(
                        "type",
                        "function",
                        "function",
                        Map.of(
                                "name",
                                "get_weather",
                                "description",
                                "Current weather",
                                "parameters",
                                parameters)),
                tool.definition());
    }

    @Test
    void absentOptionalValuesProduceAParameterlessFunction() {
        Tool tool = Tool.of("noop", null, null);
        Map<?, ?> function = (Map<?, ?>) tool.definition().get("function");

        assertFalse(function.containsKey("description"));
        assertEquals(Map.of("type", "object"), tool.parameters());
        assertThrows(IllegalArgumentException.class, () -> Tool.of("", null, null));
    }
}
