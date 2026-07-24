// Oracle: NemotronHTurnTemplate's tool encoding (system-turn declarations + instructions, call
// turns with history-thinking truncation, tool-response folding) must be token-exact with the
// GGUF's own chat_template rendered with tools.
package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.testkit.CodecOracleScenario;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public final class NemotronToolOracle {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    // enum, default, integer, boolean, string array, nested object: exercises the handled keys
    // AND render_extra_keys (items/properties/default fall through as extra keys)
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
        Path model =
                Path.of(
                        "/home/mukel/Desktop/playground/models/bartowski/"
                                + "nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf");
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(model, NemotronHTurnTemplate::new, Map.of());

        o.compareTools(
                "tools, no system (default persona)",
                List.of(WEATHER),
                List.of(Message.user("Weather in Paris?")));
        o.compareTools(
                "system + tools",
                List.of(WEATHER),
                List.of(
                        Message.system("You are a concise assistant."),
                        Message.user("Weather in Paris?")));
        o.compareTools(
                "rich schema (enum, defaults, array items, nested object via extra keys)",
                List.of(WEATHER, SEARCH),
                List.of(Message.user("Find rivers.")));

        o.compareTools(
                "call turn (multi-arg, typed values)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", java.util.Map.of("city", "Paris"))));
        o.compareTools(
                "call + tool result (response folded into a user turn)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));
        o.compareTools(
                "answered round-trip + next question (call before last user: truncated)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.Text("Let me check."),
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris")))),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny"))),
                        Message.assistant("It is 18C and sunny."),
                        Message.user("And tomorrow?")));
        o.compareTools(
                "two consecutive tool results share one user turn",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris and Rome?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C"))),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "24C")))));

        o.finish("NemotronToolOracle");
    }
}
