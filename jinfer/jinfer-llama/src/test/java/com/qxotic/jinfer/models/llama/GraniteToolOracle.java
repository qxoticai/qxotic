// Oracle: GraniteTurnTemplate's tool encoding (tools message in the system turn, JSON call
// blocks, tool-response folding into one user turn) must be token-exact with the GGUF's own
// chat_template rendered with tools.
package com.qxotic.jinfer.models.llama;

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

public final class GraniteToolOracle {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    static final Tool SEARCH =
            new Tool(
                    "search",
                    "{\"type\": \"function\", \"function\": {\"name\": \"search\","
                            + " \"description\": \"Search the archive\", \"parameters\": {\"type\":"
                            + " \"object\", \"properties\": {\"query\": {\"type\": \"string\","
                            + " \"description\": \"What to look for\"}, \"mode\": {\"type\":"
                            + " \"string\", \"enum\": [\"fast\", \"deep\"]}, \"limit\": {\"type\":"
                            + " \"integer\"}}, \"required\": [\"query\"]}}}");

    static Message assistantCall(String name, Map<String, Object> args) {
        return new Message(Role.ASSISTANT, List.of(new Part.ToolCall("", name, args)));
    }

    @Test
    void oracle() throws Exception {
        Path model =
                Path.of(
                        "/home/mukel/Desktop/playground/models/ibm-granite/granite-4.1-3b-Q8_0.gguf");
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o = new CodecOracleScenario(model, GraniteTurnTemplate::new, Map.of());

        o.compareTools(
                "tools, no system", List.of(WEATHER), List.of(Message.user("Weather in Paris?")));
        o.compareTools(
                "system + tools",
                List.of(WEATHER, SEARCH),
                List.of(
                        Message.system("You are a concise assistant."),
                        Message.user("Weather in Paris?")));
        o.compareTools(
                "call turn",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris"))));
        o.compareTools(
                "content + call in one turn",
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
                "call + result + answer + next question",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
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

        o.finish("GraniteToolOracle");
    }
}
