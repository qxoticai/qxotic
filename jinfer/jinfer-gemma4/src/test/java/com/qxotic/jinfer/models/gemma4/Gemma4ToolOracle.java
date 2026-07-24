// Oracle: Gemma4TurnTemplate's tool encoding (declarations in the system turn, the one-open-model-
// turn call round-trip) must be token-exact with the GGUF's own chat_template rendered with tools.
package com.qxotic.jinfer.models.gemma4;

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

public final class Gemma4ToolOracle {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\","
                        + " \"description\": \"Get current weather for a city\", \"parameters\":"
                        + " {\"type\": \"object\", \"properties\": {\"city\": {\"type\":"
                        + " \"string\"}}, \"required\": [\"city\"]}}}");

    // the byte-exact declaration block the template's macros produce for WEATHER
    static final String WEATHER_DECL =
            "declaration:get_weather{description:<|\"|>Get current weather for a"
                    + " city<|\"|>,parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},"
                    + "required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}";

    static final String SYS = "<bos><|turn>system\n<|tool>" + WEATHER_DECL + "<tool|><turn|>\n";
    static final String GEN = "<|turn>model\n";

    static Message assistantCall(String name, Map<String, Object> args) {
        return new Message(Role.ASSISTANT, List.of(new Part.ToolCall("", name, args)));
    }

    @Test
    void oracle() throws Exception {
        Path model =
                Path.of("/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-Q8_0.gguf");
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(
                        model, Gemma4TurnTemplate::new, Map.of("bos_token", "<bos>"));

        // Everything is validated against known-correct rendered strings: jinfer-jinja cannot
        // evaluate the template's macros (dictsort/namespace), so the strings ARE the oracle.
        o.compareToolsExpected(
                "tools, no system",
                SYS + "<|turn>user\nWeather in Paris?<turn|>\n" + GEN,
                List.of(WEATHER),
                List.of(Message.user("Weather in Paris?")));
        o.compareToolsExpected(
                "system + tools",
                "<bos><|turn>system\nYou are a concise assistant.<|tool>"
                        + WEATHER_DECL
                        + "<tool|><turn|>\n<|turn>user\nWeather in Paris?<turn|>\n"
                        + GEN,
                List.of(WEATHER),
                List.of(
                        Message.system("You are a concise assistant."),
                        Message.user("Weather in Paris?")));

        // Call/response flow against known-correct rendered strings (the one open model turn).
        o.compareToolsExpected(
                "trailing call turn (awaiting results)",
                SYS
                        + "<|turn>user\nWeather in Paris?<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                        + "<|tool_response>",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris"))));

        o.compareToolsExpected(
                "call + result (generate the answer in the open turn)",
                SYS
                        + "<|turn>user\nWeather in Paris?<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                        + "<|tool_response>response:get_weather{value:<|\"|>18C, sunny<|\"|>}"
                        + "<tool_response|>",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));

        o.compareToolsExpected(
                "answered round-trip + next question",
                SYS
                        + "<|turn>user\nWeather in Paris?<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                        + "<|tool_response>response:get_weather{value:<|\"|>18C, sunny<|\"|>}"
                        + "<tool_response|>"
                        + "It is 18C and sunny.<turn|>\n"
                        + "<|turn>user\nAnd tomorrow?<turn|>\n"
                        + GEN,
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny"))),
                        Message.assistant("It is 18C and sunny."),
                        Message.user("And tomorrow?")));

        // numeric + dictsorted args: template dictsorts call arguments (b before top_k? no:
        // case-insensitive key order), numbers render bare
        var args = new java.util.LinkedHashMap<String, Object>();
        args.put("top_k", 3L);
        args.put("q", "rivers"); // insertion order q AFTER top_k: dictsort must emit q first
        o.compareToolsExpected(
                "dictsorted numeric args",
                SYS
                        + "<|turn>user\nsearch<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{q:<|\"|>rivers<|\"|>,top_k:3}<tool_call|>"
                        + "<|tool_response>",
                List.of(WEATHER),
                List.of(Message.user("search"), assistantCall("get_weather", args)));

        o.finish("Gemma4ToolOracle");
    }
}
