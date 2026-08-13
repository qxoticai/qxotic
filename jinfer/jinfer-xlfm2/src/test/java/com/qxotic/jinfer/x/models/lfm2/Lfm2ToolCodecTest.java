package com.qxotic.jinfer.x.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.Tool;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class Lfm2ToolCodecTest {

    @Test
    void pythonCallsRoundTripEveryJsonValueShape() {
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("stars", 4L);
        nested.put("tags", List.of("pool", "spa"));
        Map<String, Object> arguments = new LinkedHashMap<>();
        arguments.put("text", "it's \\fine\nnow");
        arguments.put("count", 3L);
        arguments.put("ratio", 2.5);
        arguments.put("enabled", true);
        arguments.put("missing", null);
        arguments.put("filters", nested);

        String wire =
                Lfm2ToolCodec.renderCalls(
                        List.of(new Content.ToolCall("", "configure", arguments)));
        List<Content.ToolCall> parsed = Lfm2ToolCodec.parse(wire);

        assertEquals(1, parsed.size());
        assertEquals("configure", parsed.getFirst().name());
        assertEquals(arguments, parsed.getFirst().arguments());
        assertTrue(wire.startsWith("[configure(text='it\\'s \\\\fine\\nnow'"), wire);
    }

    @Test
    void acceptsMultipleBareAndJsonCalls() {
        assertEquals(
                List.of("first", "second"),
                Lfm2ToolCodec.parse("[first(x=1), second(items=('a', 'b'))]").stream()
                        .map(Content.ToolCall::name)
                        .toList());
        assertEquals("bare", Lfm2ToolCodec.parse("bare(flag=True)").getFirst().name());

        String json =
                "[{\"name\":\"one\",\"arguments\":{\"n\":2}},"
                        + "{\"function\":{\"name\":\"two\",\"arguments\":\"{\\\"ok\\\":true}\"}}]";
        List<Content.ToolCall> calls = Lfm2ToolCodec.parse(json);
        assertEquals(List.of("one", "two"), calls.stream().map(Content.ToolCall::name).toList());
        assertEquals(2L, calls.getFirst().arguments().get("n"));
        assertEquals(true, calls.getLast().arguments().get("ok"));

        Content.ToolCall spaced =
                Lfm2ToolCodec.parse("[ {\"name\":\"three\",\"parameters\":{\"n\":3}} ]").getFirst();
        assertEquals(3L, spaced.arguments().get("n"));
    }

    @Test
    void toleratesObservedUnescapedInnerQuotesAndRejectsGarbage() {
        List<Content.ToolCall> calls =
                Lfm2ToolCodec.parse("[send(text=\"He said \"hello\" today.\")]");
        assertEquals("He said \"hello\" today.", calls.getFirst().arguments().get("text"));
        assertEquals(
                Map.of("key", "value"),
                Lfm2ToolCodec.parse("f(options={'key': 'value',})")
                        .getFirst()
                        .arguments()
                        .get("options"));
        assertTrue(Lfm2ToolCodec.parse("[not a( valid ]call").isEmpty());
    }

    @Test
    void toolDefinitionsUseJinjaJsonSpacingAndInsertionOrder() {
        Map<String, Object> definition = new LinkedHashMap<>();
        definition.put("name", "weather");
        definition.put("description", "Weather in \"city\"");
        definition.put("parameters", Map.of("type", "object"));

        assertEquals(
                "List of tools: [{\"name\": \"weather\", \"description\": "
                        + "\"Weather in \\\"city\\\"\", \"parameters\": {\"type\": \"object\"}}]",
                Lfm2ToolCodec.renderTools(List.of(new Tool("weather", definition))));
    }
}
