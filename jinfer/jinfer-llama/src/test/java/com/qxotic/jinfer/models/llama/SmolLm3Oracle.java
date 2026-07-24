// Oracle: SmolLm3ChatTemplate must be token-exact with the GGUF's own Jinja chat_template - the
// metadata system header, /think / /no_think modes, python-repr tool signatures. The call-history
// case uses an expected string (the template ignores tool_calls; the port's rendering is a
// documented deviation).
package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.testkit.CodecOracleScenario;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public final class SmolLm3Oracle {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    @Test
    void oracle() throws Exception {
        Path model = Path.of("/home/mukel/Desktop/playground/models/ggml-org/SmolLM3-Q4_K_M.gguf");
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        String today =
                LocalDate.now().format(DateTimeFormatter.ofPattern("dd MMMM yyyy", Locale.ENGLISH));
        CodecOracleScenario o =
                new CodecOracleScenario(model, tk -> new SmolLm3ChatTemplate(tk, today), Map.of());

        o.compare("single user", List.of(Message.user("What is the capital of France?")));
        o.compare(
                "system + user (custom instructions)",
                List.of(
                        Message.system("You are a concise assistant."),
                        Message.user("Give me a haiku.")));
        o.compare(
                "no-think mode",
                false,
                Map.of("enable_thinking", false),
                List.of(Message.user("Hi!")));
        o.compare(
                "/no_think switch in the system message",
                List.of(Message.system("Be brief. /no_think"), Message.user("Hi!")));
        o.compare(
                "multi-turn",
                List.of(
                        Message.user("Hi!"),
                        Message.assistant("Hello! How can I help?"),
                        Message.user("Name three primes.")));
        o.compareTools(
                "tools (python-repr signatures)",
                List.of(WEATHER),
                List.of(Message.user("Weather in Paris?")));
        o.compareTools(
                "tools + tool result as a user turn",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        Message.assistant("Checking."),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));

        // the documented deviation: the template drops tool_calls from history; the port renders
        // them as the model emits them, so the string IS the oracle for this shape
        String header =
                "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: "
                        + today
                        + "\nReasoning Mode: /think\n\n## Custom Instructions\n\n"
                        + SmolLm3ChatTemplate.DEFAULT_THINK_PERSONA
                        + "### Tools\n\n"
                        + SmolLm3ChatTemplate.TOOLS_INTRO
                        + "{'type': 'function', 'function': {'name': 'get_weather',"
                        + " 'description': 'Get current weather for a city', 'parameters':"
                        + " {'type': 'object', 'properties': {'city': {'type': 'string'}},"
                        + " 'required': ['city']}}}\n"
                        + SmolLm3ChatTemplate.TOOLS_OUTRO_HEAD
                        + "<tool_call></tool_call> XML tags:\n<tool_call>"
                        + SmolLm3ChatTemplate.TOOLS_EXAMPLE_BODY
                        + "</tool_call>\n\n<|im_end|>\n";
        o.compareToolsExpected(
                "call turn renders as the model emits it (deviation)",
                header
                        + "<|im_start|>user\nWeather in Paris?<|im_end|>\n"
                        + "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\","
                        + " \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|im_end|>\n"
                        + "<|im_start|>assistant\n",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris"))))));

        o.finish("SmolLm3Oracle");
    }
}
