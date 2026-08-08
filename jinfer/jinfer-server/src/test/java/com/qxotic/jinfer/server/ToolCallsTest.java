package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Parsing a model's reply into OpenAI tool calls. Model-free: these are string shapes, and the
 * battery that used to be the only thing covering them needs a GGUF and does not run in a normal
 * build.
 */
final class ToolCallsTest {

    private static Map<String, Object> one(String text, String... knownTools) {
        List<Map<String, Object>> calls = ToolCalls.parseToolCalls(text, Set.of(knownTools));
        assertEquals(1, calls.size(), "expected exactly one call from: " + text);
        return calls.get(0);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> function(Map<String, Object> call) {
        return (Map<String, Object>) call.get("function");
    }

    /**
     * Meta's Llama 3.x prompt asks for {@code parameters} in so many words - "Respond in the format
     * {"name": function name, "parameters": dictionary of argument name and its value}" - so that
     * is the key those models emit. The recognizer already accepted either spelling when deciding a
     * blob WAS a call; the normalizer only ever read {@code arguments}, so every Llama 3.x call
     * arrived named correctly with its arguments silently replaced by {}.
     */
    @Test
    void argumentsMayBeSpelledParameters() {
        Map<String, Object> call =
                one("{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}");
        assertEquals("get_weather", function(call).get("name"));
        String arguments = (String) function(call).get("arguments");
        assertTrue(arguments.contains("Paris"), "the arguments must survive: " + arguments);
        assertFalse("{}".equals(arguments), "an emptied argument object is the bug");
    }

    /** The same, with the "type": "function" key Llama adds when it has seen an OpenAI schema. */
    @Test
    void aTypedCallWithParametersParsesToo() {
        Map<String, Object> call =
                one(
                        "<|python_tag|>{\"type\": \"function\", \"name\": \"get_weather\","
                                + " \"parameters\": {\"city\": \"Paris\"}}");
        assertEquals("get_weather", function(call).get("name"));
        assertTrue(((String) function(call).get("arguments")).contains("Paris"));
    }

    /** OpenAI's own spelling keeps working, nested and flat. */
    @Test
    void openAiSpellingsStillParse() {
        Map<String, Object> flat =
                one("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}");
        assertTrue(((String) function(flat).get("arguments")).contains("Rome"));

        Map<String, Object> nested =
                one(
                        "{\"tool_calls\": [{\"function\": {\"name\": \"get_weather\","
                                + " \"arguments\": {\"city\": \"Rome\"}}}]}");
        assertEquals("get_weather", function(nested).get("name"));
        assertTrue(((String) function(nested).get("arguments")).contains("Rome"));
    }

    /** arguments wins when a reply carries both spellings - it is the one OpenAI defines. */
    @Test
    void argumentsWinsOverParameters() {
        Map<String, Object> call =
                one(
                        "{\"name\": \"f\", \"arguments\": {\"a\": 1}, \"parameters\": {\"b\":"
                                + " 2}}");
        String arguments = (String) function(call).get("arguments");
        assertTrue(arguments.contains("\"a\""), arguments);
        assertFalse(arguments.contains("\"b\""), arguments);
    }

    /** Ordinary prose is not a tool call, whatever brackets it happens to contain. */
    @Test
    void proseIsNotACall() {
        assertTrue(
                ToolCalls.parseToolCalls(
                                "See [the docs](https://example.com) and call print() first.",
                                Set.of("get_weather"))
                        .isEmpty());
    }
}
