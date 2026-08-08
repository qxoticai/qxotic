// Pins the UPSTREAM chat-template fixes (google/gemma-4-26B-A4B-it commit 35b4173: "null
// handling, reasoning preservation, turn-tag balance, input validation") that the shipped GGUFs
// predate. Like Gemma4ToolOracle, the expected strings ARE the oracle (jinfer-jinja cannot
// evaluate the template's macros); each string is derived from the fixed template's fragments:
//   thinking_gate  = index > last_user_idx  ->  '<|channel>thought\n' + text + '\n<channel|>'
//   continues_into_next -> no '<turn|>' between consecutive model turns
//   close unless (responses and no content and NO next non-tool turn)
//   format_argument(none) -> 'null'
// The wire-side validation fix (arguments must be a JSON object) is a server test concern.
// The fixed template text itself is checked in for reference at
// src/test/resources/gemma4-chat-template-fixed.jinja (jinfer-jinja cannot execute its macros).
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.testkit.CodecOracleScenario;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

public final class Gemma4FixedTemplateOracle {

    static final Tool WEATHER = Gemma4ToolOracle.WEATHER;
    static final String SYS = Gemma4ToolOracle.SYS;
    static final String GEN = Gemma4ToolOracle.GEN;

    /** The Paris round-trip every tool case shares: system+user+call+folded response. */
    static final String PARIS_LOOP =
            SYS
                    + "<|turn>user\nWeather in Paris?<turn|>\n"
                    + GEN
                    + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                    + "<|tool_response>response:get_weather{value:<|\"|>18C, sunny<|\"|>}"
                    + "<tool_response|>";

    static List<Message> parisLoop(Part result) {
        return List.of(
                Message.user("Weather in Paris?"),
                Gemma4ToolOracle.assistantCall("get_weather", Map.of("city", "Paris")),
                new Message(Role.TOOL, List.of(result)));
    }

    static Message reasoningAssistant(String reasoning, Part... rest) {
        List<Part> parts = new ArrayList<>();
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
        List<Message> loopThenUser =
                new ArrayList<>(parisLoop(new Part.ToolResult("", "18C, sunny")));
        loopThenUser.add(Message.user("And Berlin?"));
        o.compareToolsExpected(
                "responses turn closes before a following user turn",
                PARIS_LOOP + "<turn|>\n<|turn>user\nAnd Berlin?<turn|>\n" + GEN,
                List.of(WEATHER),
                loopThenUser);

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

    /**
     * The tool-response thought tail (scaffolded checkpoints only, 12B/26B): after a trailing tool
     * response with thinking on, the prompt re-opens the thought channel and the reply seed IS that
     * tail - co-produced, so the two cannot disagree.
     */
    @Test
    void toolResponseThoughtTail() throws Exception {
        Path model = ModelFixture.GEMMA4_E2B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        CodecOracleScenario o =
                new CodecOracleScenario(
                        model,
                        tokenizer -> new Gemma4TurnTemplate(tokenizer, null, 0, true),
                        Map.of("bos_token", "<bos>"));
        List<Message> loop =
                List.of(
                        Message.user("Weather in Paris?"),
                        Gemma4ToolOracle.assistantCall("get_weather", Map.of("city", "Paris")),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny"))));

        // thinking on: the tail opens the channel (compareToolsExpected drives thinking=true)
        o.compareToolsExpected(
                "trailing tool response opens the thought channel",
                SYS
                        + "<|turn>user\nWeather in Paris?<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                        + "<|tool_response>response:get_weather{value:<|\"|>18C, sunny<|\"|>}"
                        + "<tool_response|><|channel>thought\n",
                List.of(WEATHER),
                loop);

        // co-production: the reply seed equals the ids the encoded prompt actually ends with
        var thinkingOn = new Conversation(loop, List.of(WEATHER), true, "");
        var prompt = o.template.encodePrompt(thinkingOn);
        int[] seed = prompt.replySeed();
        o.check(seed.length > 0, "thinking tail produces a reply seed");
        List<Integer> ids = new ArrayList<>();
        for (int id : Batch.tokenIds(prompt.batches())) ids.add(id);
        boolean endsWithSeed = ids.size() >= seed.length;
        for (int k = 0; endsWithSeed && k < seed.length; k++) {
            endsWithSeed = ids.get(ids.size() - seed.length + k) == seed[k];
        }
        o.check(endsWithSeed, "the prompt ends with exactly the reply seed");

        // thinking off: no tail, no seed - the model answers in the open turn
        var thinkingOff = new Conversation(loop, List.of(WEATHER), false, "");
        var offPrompt = o.template.encodePrompt(thinkingOff);
        o.check(offPrompt.replySeed().length == 0, "thinking off has no reply seed");
        List<Integer> offIds = new ArrayList<>();
        for (int id : Batch.tokenIds(offPrompt.batches())) offIds.add(id);
        o.check(
                offIds.get(offIds.size() - 1) == o.special("<tool_response|>"),
                "thinking off ends at the folded response");

        // llama.cpp patch: a CLOSED final turn (call + answer content in one message) reopens
        // the model turn instead of emitting a bare thought channel outside any turn
        o.compareToolsExpected(
                "closed call+content turn reopens the model turn",
                PARIS_LOOP + "Checking now.<turn|>\n" + GEN,
                List.of(WEATHER),
                List.of(
                        Message.user("Weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris")),
                                        new Part.Text("Checking now."))),
                        new Message(Role.TOOL, List.of(new Part.ToolResult("", "18C, sunny")))));

        // the SERVER's lowering shape: a tool turn carrying Part.Text (not Part.ToolResult) must
        // fold identically - dropping it silently starved the model of every served tool result
        o.compareToolsExpected(
                "Text-shaped tool turn folds as a response (server lowering)",
                PARIS_LOOP + "<|channel>thought\n",
                List.of(WEATHER),
                parisLoop(new Part.Text("18C, sunny")));

        // parallel calls with id-less (server-shaped) results fold positionally, one response
        // block per result, in call order
        o.compareToolsExpected(
                "parallel calls fold id-less results positionally",
                SYS
                        + "<|turn>user\nParis and Berlin?<turn|>\n"
                        + GEN
                        + "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>"
                        + "<|tool_call>call:get_weather{city:<|\"|>Berlin<|\"|>}<tool_call|>"
                        + "<|tool_response>response:get_weather{value:<|\"|>18C<|\"|>}"
                        + "<tool_response|>"
                        + "<|tool_response>response:get_weather{value:<|\"|>12C<|\"|>}"
                        + "<tool_response|><|channel>thought\n",
                List.of(WEATHER),
                List.of(
                        Message.user("Paris and Berlin?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Paris")),
                                        new Part.ToolCall(
                                                "", "get_weather", Map.of("city", "Berlin")))),
                        new Message(Role.TOOL, List.of(new Part.Text("18C"))),
                        new Message(Role.TOOL, List.of(new Part.Text("12C")))));

        o.finish("Gemma4FixedTemplateOracle[thoughtTail]");
    }
}
