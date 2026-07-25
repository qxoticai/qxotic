// Oracle: MistralChatTemplate must be token-exact with the GGUF's own Jinja chat_template - the
// v13 wire ([SYSTEM_PROMPT] default persona, [AVAILABLE_TOOLS] JSON array, [INST] turns, bare
// assistant continuations with per-message </s>, [TOOL_CALLS]name[ARGS]{json} calls,
// [TOOL_RESULTS] results).
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

public final class MistralOracle {

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
        Path model = ModelFixture.MINISTRAL_3B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(
                        model,
                        MistralChatTemplate::new,
                        Map.of("bos_token", "<s>", "eos_token", "</s>"));

        o.compare("single user (default persona)", List.of(Message.user("Capital of France?")));
        o.compare(
                "system + user", List.of(Message.system("You are concise."), Message.user("Hi!")));
        o.compare(
                "multi-turn (per-message </s>)",
                List.of(
                        Message.user("Hi!"),
                        Message.assistant("Hello!"),
                        Message.user("Name three primes.")));
        o.compareTools(
                "tools (one JSON array in [AVAILABLE_TOOLS])",
                List.of(WEATHER),
                List.of(Message.user("Weather in Paris?")));
        o.compareTools(
                "call turn ([TOOL_CALLS]name[ARGS]json)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris"))));
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

        o.finish("MistralOracle");
    }
}
