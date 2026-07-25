// Oracle: MiniCpm5ChatTemplate must be token-exact with the GGUF's own Jinja chat_template - the
// XML function wire (trusted <function ids, CDATA values), tojson declarations inside <tools>
// ids, tool responses folded into user turns, last-query reasoning retention.
package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.testkit.CodecOracleScenario;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public final class MiniCpm5Oracle {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    static Message assistantCall(String name, Map<String, Object> args) {
        return new Message(Role.ASSISTANT, List.of(new Part.ToolCall("", name, args)));
    }

    @Test
    void oracle() throws Exception {
        Path model = ModelFixture.MINICPM5_1B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(
                        model, MiniCpm5ChatTemplate::new, Map.of("bos_token", "<s>"));

        o.compare(
                "single user (thinking prompt)",
                true,
                Map.of("enable_thinking", true),
                List.of(Message.user("What is the capital of France?")));
        o.compare(
                "system + user, no thinking",
                false,
                Map.of("enable_thinking", false),
                List.of(Message.system("You are concise."), Message.user("Hi!")));
        o.compare(
                "multi-turn",
                true,
                Map.of("enable_thinking", true),
                List.of(
                        Message.user("Hi!"),
                        Message.assistant("Hello!"),
                        Message.user("Name three primes.")));
        o.compareTools(
                "tools declarations (trusted <tools>/<function> ids)",
                true,
                Map.of("enable_thinking", true),
                List.of(WEATHER),
                List.of(Message.user("Weather in Paris?")));
        o.compareTools(
                "call turn (XML function wire)",
                true,
                Map.of("enable_thinking", true),
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris"))));
        o.compareTools(
                "CDATA value (newline in a param)",
                true,
                Map.of("enable_thinking", true),
                List.of(WEATHER),
                List.of(
                        Message.user("Weather?"),
                        assistantCall("get_weather", Map.of("city", "Paris\nFrance"))));
        o.compareTools(
                "call + tool result folded into a user turn",
                true,
                Map.of("enable_thinking", true),
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));
        o.compareTools(
                "answered round-trip + next question",
                true,
                Map.of("enable_thinking", true),
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C"))),
                        Message.assistant("It is 18C."),
                        Message.user("And tomorrow?")));

        o.finish("MiniCpm5Oracle");
    }
}
