// Pins the UPSTREAM chat-template fixes (google/gemma-4-26B-A4B-it commit 35b4173: "null
// handling, reasoning preservation, turn-tag balance, input validation") that the shipped GGUFs
// predate. Like Gemma4ToolOracle, the expected strings ARE the oracle (jinfer-jinja cannot
// evaluate the template's macros); each string is derived from the fixed template's fragments:
//   thinking_gate  = index > last_user_idx  ->  '<|channel>thought\n' + text + '\n<channel|>'
//   continues_into_next -> no '<turn|>' between consecutive model turns
//   close unless (responses and no content and NO next non-tool turn)
//   format_argument(none) -> 'null'
// The wire-side validation fix (arguments must be a JSON object) is a server test concern.
package com.qxotic.jinfer.models.gemma4;

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

public final class Gemma4FixedTemplateOracle {

    static final Tool WEATHER = Gemma4ToolOracle.WEATHER;
    static final String SYS = Gemma4ToolOracle.SYS;
    static final String GEN = Gemma4ToolOracle.GEN;

    static Message reasoningAssistant(String reasoning, Part... rest) {
        List<Part> parts = new java.util.ArrayList<>();
        parts.add(new Part.Reasoning(List.of(new Part.Text(reasoning)), null));
        parts.addAll(List.of(rest));
        return new Message(Role.ASSISTANT, parts);
    }

    @Test
    void oracle() throws Exception {
        Path model = ModelFixture.GEMMA4_E2B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(
                        model, Gemma4TurnTemplate::new, Map.of("bos_token", "<bos>"));

        // reasoning STRIPPED: the assistant turn sits before the last user message
        o.compareToolsExpected(
                "reasoning stripped before the last user turn",
                "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n"
                        + "<|turn>user\nBye<turn|>\n"
                        + GEN,
                List.of(),
                List.of(
                        Message.user("Hi"),
                        reasoningAssistant("the user greets me", new Part.Text("Hello!")),
                        Message.user("Bye")));

        // reasoning PRESERVED: the tool-loop turn is after the last user message
        o.compareToolsExpected(
                "reasoning preserved in the tool loop",
                SYS
                        + "<|turn>user\nWeather in Paris?<turn|>\n"
                        + GEN
                        + "<|channel>thought\nI should check the weather tool.\n<channel|>"
                        + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                        + "<|tool_response>response:get_weather{value:<|\"|>18C, sunny<|\"|>}"
                        + "<tool_response|>",
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        reasoningAssistant(
                                "I should check the weather tool.",
                                new Part.ToolCall("", "get_weather", Map.of("city", "Paris"))),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));

        // turn-tag balance: consecutive assistant messages merge into ONE model turn
        o.compareToolsExpected(
                "consecutive assistant turns merge",
                "<bos><|turn>user\nQ<turn|>\n<|turn>model\nAB<turn|>\n" + GEN,
                List.of(),
                List.of(Message.user("Q"), Message.assistant("A"), Message.assistant("B")));

        // turn-tag balance: a folded-responses turn with no answer CLOSES when a next turn
        // follows (previously it stayed open and swallowed the following user turn)
        o.compareToolsExpected(
                "responses turn closes before a following user turn",
                SYS
                        + "<|turn>user\nWeather in Paris?<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                        + "<|tool_response>response:get_weather{value:<|\"|>18C, sunny<|\"|>}"
                        + "<tool_response|><turn|>\n"
                        + "<|turn>user\nAnd Berlin?<turn|>\n"
                        + GEN,
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        Gemma4ToolOracle.assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny"))),
                        Message.user("And Berlin?")));

        // null handling: a null argument renders as JSON null, not Python None
        var nullArg = new LinkedHashMap<String, Object>();
        nullArg.put("city", null);
        o.compareToolsExpected(
                "null argument renders 'null'",
                SYS
                        + "<|turn>user\nsearch<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{city:null}<tool_call|>"
                        + "<|tool_response>",
                List.of(WEATHER),
                List.of(
                        Message.user("search"),
                        Gemma4ToolOracle.assistantCall("get_weather", nullArg)));

        o.finish("Gemma4FixedTemplateOracle");
    }
}
