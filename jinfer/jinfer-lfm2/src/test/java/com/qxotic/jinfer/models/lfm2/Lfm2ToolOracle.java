// Oracle: Lfm2TurnTemplate's tool encoding (tools in the system turn, assistant tool-call turns,
// tool-result turns) must be token-exact with the GGUF's own Jinja chat_template rendered with a
// `tools` var.   java ... com.qxotic.jinfer.models.lfm2.Lfm2ToolOracle [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.testkit.OracleScenario;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public final class Lfm2ToolOracle {

    // rawJson is in jinja tojson canonical form (", "/": " separators, insertion order), which is
    // what the template's `tool | tojson` produces and what the server must canonicalize to.
    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "Get current weather for a city",
                    "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}},"
                            + " \"required\": [\"city\"]}",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    static final Tool SEARCH =
            new Tool(
                    "web_search",
                    "Search the web",
                    "{\"type\": \"object\", \"properties\": {\"q\": {\"type\": \"string\"},"
                            + " \"top_k\": {\"type\": \"integer\"}}}",
                    "{\"type\": \"function\", \"function\": {\"name\": \"web_search\","
                        + " \"description\": \"Search the web\", \"parameters\": {\"type\":"
                        + " \"object\", \"properties\": {\"q\": {\"type\": \"string\"}, \"top_k\":"
                        + " {\"type\": \"integer\"}}}}}");

    static Message assistantCall(String name, Map<String, Object> args) {
        return new Message(Role.ASSISTANT, List.of(new Part.ToolCall("", name, args)));
    }

    static Message toolResult(String content) {
        return new Message(Role.TOOL, content);
    }

    public static void main(String[] args) throws Exception {
        Path model =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        if (!Files.exists(model)) {
            System.out.println("Lfm2ToolOracle: model not found (" + model + "), skipping");
            return;
        }
        OracleScenario o =
                new OracleScenario(
                        model,
                        Lfm2TurnTemplate::new,
                        Map.of("bos_token", "<|startoftext|>", "eos_token", "<|im_end|>"));

        o.compareTools(
                "tools, no system",
                true,
                List.of(WEATHER),
                List.of(Message.user("Weather in Paris?")));

        o.compareTools(
                "system + tools",
                true,
                List.of(WEATHER),
                List.of(
                        Message.system("You are a concise assistant."),
                        Message.user("Weather in Paris?")));

        o.compareTools("two tools", true, List.of(WEATHER, SEARCH), List.of(Message.user("Hi")));

        // Tool-CALL turns are validated against the known-correct rendered string: jinfer-jinja
        // cannot evaluate LFM2's render_tool_calls macro (see OracleScenario.compareToolsExpected).
        String sys =
                "<|startoftext|><|im_start|>system\nList of tools: ["
                        + WEATHER.rawJson()
                        + "]<|im_end|>\n";
        o.compareToolsExpected(
                "assistant tool call",
                sys
                        + "<|im_start|>user\n"
                        + "Weather in Paris?<|im_end|>\n"
                        + "<|im_start|>assistant\n"
                        + "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|><|im_end|>\n",
                false,
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris"))));

        o.compareToolsExpected(
                "full round-trip",
                sys
                        + "<|im_start|>user\n"
                        + "Weather in Paris?<|im_end|>\n"
                        + "<|im_start|>assistant\n"
                        + "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|><|im_end|>\n"
                        + "<|im_start|>tool\n"
                        + "18C, sunny<|im_end|>\n"
                        + "<|im_start|>assistant\n",
                true,
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        toolResult("18C, sunny")));

        var searchArgs = new java.util.LinkedHashMap<String, Object>();
        searchArgs.put("q", "rivers");
        searchArgs.put("top_k", 3L);
        o.compareToolsExpected(
                "numeric + string args",
                "<|startoftext|><|im_start|>system\nList of tools: ["
                        + SEARCH.rawJson()
                        + "]<|im_end|>\n"
                        + "<|im_start|>user\n"
                        + "search<|im_end|>\n"
                        + "<|im_start|>assistant\n"
                        + "<|tool_call_start|>[web_search(q='rivers',"
                        + " top_k=3)]<|tool_call_end|><|im_end|>\n",
                false,
                List.of(SEARCH),
                List.of(Message.user("search"), assistantCall("web_search", searchArgs)));

        o.finish("Lfm2ToolOracle");
    }
}
