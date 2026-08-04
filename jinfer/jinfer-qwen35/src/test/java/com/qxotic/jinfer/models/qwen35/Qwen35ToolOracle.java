// Oracle: Qwen35TurnTemplate's tool encoding (system-turn declarations + instructions, XML
// function call turns, tool-response folding, last-real-query thinking policy) must be
// token-exact with the GGUF's own chat_template rendered with tools.
package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.testkit.CodecOracleScenario;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public final class Qwen35ToolOracle {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    // enum, default, integer, boolean, string array, nested object: the declarations block is a
    // raw tojson of the whole tool object, so the schema passes through verbatim
    static final Tool SEARCH =
            new Tool(
                    "search",
                    "{\"type\": \"function\", \"function\": {\"name\": \"search\","
                            + " \"description\": \"Search the archive\", \"parameters\": {\"type\":"
                            + " \"object\", \"properties\": {\"query\": {\"type\": \"string\","
                            + " \"description\": \"What to look for\"}, \"mode\": {\"type\":"
                            + " \"string\", \"enum\": [\"fast\", \"deep\"], \"default\": \"fast\"},"
                            + " \"limit\": {\"type\": \"integer\", \"default\": 10}, \"fuzzy\":"
                            + " {\"type\": \"boolean\"}, \"tags\": {\"type\": \"array\", \"items\":"
                            + " {\"type\": \"string\"}}, \"filters\": {\"type\": \"object\","
                            + " \"properties\": {\"stars\": {\"type\": \"integer\"}}}},"
                            + " \"required\": [\"query\"]}}}");

    static Message assistantCall(String name, Map<String, Object> args) {
        return new Message(Role.ASSISTANT, List.of(new Part.ToolCall("", name, args)));
    }

    @Test
    void oracle() throws Exception {
        Path model = ModelFixture.QWEN35_2B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(
                        model, Qwen35TurnTemplate::new, Map.of("enable_thinking", true));

        o.compareTools(
                "tools, no system", List.of(WEATHER), List.of(Message.user("Weather in Paris?")));
        o.compareTools(
                "system + tools (system content appended AFTER the instructions)",
                List.of(WEATHER),
                List.of(
                        Message.system("You are a concise assistant."),
                        Message.user("Weather in Paris?")));
        o.compareTools(
                "rich schema passes through tojson verbatim",
                List.of(WEATHER, SEARCH),
                List.of(Message.user("Find rivers.")));

        o.compareTools(
                "call turn",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris"))));
        o.compareTools(
                "call with typed + structured args (tojson for maps/lists, string otherwise)",
                List.of(SEARCH),
                List.of(Message.user("Find rivers."), assistantCall("search", orderedArgs())));
        o.compareTools(
                "call + tool result (response folded into a user turn)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));
        o.compareTools(
                "content + call in one turn (\\n\\n separator)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.Text("Let me check."),
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris"))))));
        o.compareTools(
                "parallel calls in one turn (\\n separator between blocks)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris and Rome?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris")),
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Rome"))))));
        o.compareTools(
                "two consecutive tool results share one user turn",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris and Rome?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C"))),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "24C")))));
        o.compareTools(
                "answered round-trip + next question (historical thinking dropped entirely)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny"))),
                        Message.assistant("<think>\nsunny then\n</think>\n\nIt is 18C and sunny."),
                        Message.user("And tomorrow?")));
        o.compareTools(
                "assistant call after the last query keeps its reasoning",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.Reasoning(
                                                List.of(
                                                        new Part.Text(
                                                                "I should check the weather.")),
                                                null),
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris")))),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C")))));

        o.finish("Qwen35ToolOracle");
    }

    /** Deterministic arg order: the template iterates the mapping in insertion order. */
    private static Map<String, Object> orderedArgs() {
        Map<String, Object> args = new LinkedHashMap<>();
        args.put("query", "rivers");
        args.put("limit", 3);
        args.put("fuzzy", true);
        args.put("tags", List.of("nature", "water"));
        args.put("filters", Map.of("stars", 5));
        return args;
    }

    /**
     * The server lowers {@code {role:"tool", content}} to a plain Text part: both wire shapes must
     * render identically on the native path (the Nemotron regression, pinned here too).
     */
    @Test
    void serverTextShapeRendersLikeTypedToolResult() throws Exception {
        Path model = ModelFixture.QWEN35_2B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(
                        model, Qwen35TurnTemplate::new, Map.of("enable_thinking", true));
        String result = "{\"temp_c\": 7, \"condition\": \"light rain\"}";
        List<Message> head =
                List.of(
                        Message.user("Weather in Zurich?"),
                        assistantCall("get_weather", Map.of("city", "Zurich")));
        List<Integer> typed =
                o.encodeIds(
                        new Conversation(
                                concat(
                                        head,
                                        new Message(
                                                Role.TOOL,
                                                List.of(new Part.ToolResult("call_0", result)))),
                                List.of(WEATHER),
                                true,
                                ""));
        List<Integer> text =
                o.encodeIds(
                        new Conversation(
                                concat(head, new Message(new Role("tool"), result)),
                                List.of(WEATHER),
                                true,
                                ""));
        assertEquals(typed, text, "Text-shaped tool results must render like typed ToolResults");
    }

    private static List<Message> concat(List<Message> head, Message tail) {
        List<Message> out = new java.util.ArrayList<>(head);
        out.add(tail);
        return out;
    }
}
