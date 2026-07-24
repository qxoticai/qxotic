// Oracle: GptOssTurnTemplate's tool encoding (preamble routing line, developer # Tools namespace,
// call/analysis/response turns) must be token-exact with the GGUF's own chat_template rendered
// with tools. The Reasoning-retention case uses an expected string (the oracle message map cannot
// carry a thinking field).
package com.qxotic.jinfer.models.gptoss;

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
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public final class GptOssToolOracle {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    // exercises the tsType branches: enum, optional+default, integer, boolean, string array,
    // parameter descriptions
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
                            + " {\"type\": \"string\"}}}, \"required\": [\"query\"]}}}");

    static Message assistantCall(String name, Map<String, Object> args) {
        return new Message(Role.ASSISTANT, List.of(new Part.ToolCall("", name, args)));
    }

    @Test
    void oracle() throws Exception {
        Path model = Path.of("/home/mukel/Desktop/playground/models/unsloth/gpt-oss-20b-Q8_0.gguf");
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        String today = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        CodecOracleScenario o =
                new CodecOracleScenario(model, tk -> new GptOssTurnTemplate(tk, today), Map.of());

        o.compareTools(
                "tools, no system", List.of(WEATHER), List.of(Message.user("Weather in Paris?")));
        o.compareTools(
                "system + tools (# Instructions then # Tools)",
                List.of(WEATHER),
                List.of(
                        Message.system("You are a concise assistant."),
                        Message.user("Weather in Paris?")));
        o.compareTools(
                "rich schema (enum, defaults, integer, boolean, array, descriptions)",
                List.of(WEATHER, SEARCH),
                List.of(Message.user("Find rivers.")));

        o.compareTools(
                "trailing call turn (awaiting results)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris"))));
        o.compareTools(
                "call + result (generate the answer)",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));
        o.compareTools(
                "answered round-trip + next question (call analysis dropped)",
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
                "unanswered call keeps its commentary preamble as analysis",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.Text("Let me check."),
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris"))))));

        // Reasoning-part retention: the echoed reply's analysis precedes its unanswered call.
        // The Jinja-side message map cannot carry thinking, so the string IS the oracle.
        String preamble =
                "<|start|>system<|message|>"
                        + GptOssTurnTemplate.DEFAULT_IDENTITY
                        + "\nKnowledge cutoff: 2024-06\nCurrent date: "
                        + today
                        + "\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final."
                        + " Channel must be included for every message."
                        + GptOssTurnTemplate.TOOLS_LINE
                        + "<|end|>";
        String weatherNamespace =
                "## functions\n\nnamespace functions {\n\n// Get current weather for a city\ntype"
                        + " get_weather = (_: {\ncity: string,\n}) => any;\n\n} // namespace"
                        + " functions";
        o.compareToolsExpected(
                "thinking retained before an unanswered call",
                preamble
                        + "<|start|>developer<|message|># Tools\n\n"
                        + weatherNamespace
                        + "<|end|>"
                        + "<|start|>user<|message|>Weather in Paris?<|end|>"
                        + "<|start|>assistant<|channel|>analysis<|message|>Need the tool.<|end|>"
                        + "<|start|>assistant to=functions.get_weather<|channel|>commentary"
                        + " json<|message|>{\"city\": \"Paris\"}<|call|>"
                        + "<|start|>assistant",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.Reasoning(
                                                List.of(new Part.Text("Need the tool.")), null),
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris"))))));

        o.finish("GptOssToolOracle");
    }
}
