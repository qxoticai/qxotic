package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * PROTOTYPE of tools + JSON-schema response format as ONE reply-language selection: the family's
 * auto tree with the CONTENT hole carrying the schema's grammar - think stays free, calls stay the
 * family's own syntax, and visible text can only be schema JSON. The walk masks generation AND
 * parses the reply; round 1 must still CALL the tool, round 2's answer must conform. What the
 * provider today rejects loudly, expressed as a selection - the ship path is a ChatTemplate hook
 * composing this tree and prepare() wiring where the rejection sits.
 */
@Tag("integration")
class ToolsWithSchemaPrototypeIT {

    static final Map<String, Object> SCHEMA =
            Map.of(
                    "type", "object",
                    "properties",
                            Map.of(
                                    "city", Map.of("type", "string"),
                                    "temperature_c", Map.of("type", "number")),
                    "required", List.of("city", "temperature_c"));

    /** The LFM2.5 auto tree, content schema-bound: the spans preset with the hole STATED. */
    static ReplyLanguage.Node constrainedAuto() {
        return ReplyLanguage.spans(
                "<think>",
                "</think>",
                "<|tool_call_start|>",
                "<|tool_call_end|>",
                ToolCallSyntax::parseBlock,
                ReplyLanguage.mark("<|im_end|>"),
                ReplyLanguage.gbnf(Grammar.schemaGbnf(SCHEMA)));
    }

    /** One request, one walk: it masks the decode, parses the reply, and ends the pass. */
    static Message drive(JinferChatModel model, ChatRequest request) {
        ChatEngine.Prepared p = model.prepare(request);
        ReplyLanguage.Walk walk =
                ReplyLanguage.Selection.of(constrainedAuto(), model.engine.loaded().tokenizer())
                        .walk();
        for (int t : p.parserSeed()) walk.feed(t);
        walk.beginReply();
        Sampler masked =
                walk.sampler(
                        FloatTensor::argmax, model.engine.loaded().stopTokens().iterator().next());
        model.engine.generate(p.encoded().prompt(), masked, 512, 0, token -> !walk.ended());
        return walk.finish();
    }

    @Test
    void oneSelectionServesBothToolCallsAndSchemaAnswers() {
        ToolSpecification weather =
                ToolSpecification.builder()
                        .name("get_weather")
                        .description("Current weather for a city")
                        .parameters(
                                JsonObjectSchema.builder()
                                        .addStringProperty("city")
                                        .required("city")
                                        .build())
                        .build();
        try (JinferChatModel model =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.LFM25_8B_Q8.require())
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        .temperature(0.0)
                        .seed(7L)
                        .build()) {
            // the schema is SAID too (the stating law): the mask shapes, the prompt means
            UserMessage user =
                    UserMessage.from(
                            "What is the weather in Munich right now? Final answers must be JSON"
                                    + " with keys city and temperature_c, and nothing else.");
            ChatRequestParameters withTool =
                    ChatRequestParameters.builder().toolSpecifications(weather).build();
            Message reply1 =
                    drive(model, ChatRequest.builder().messages(user).parameters(withTool).build());
            List<Part.ToolCall> calls =
                    reply1.content().stream()
                            .filter(part -> part instanceof Part.ToolCall)
                            .map(part -> (Part.ToolCall) part)
                            .toList();
            assertEquals(1, calls.size(), "the schema mask must not trap the call: " + reply1);
            assertEquals("get_weather", calls.get(0).name());
            assertEquals("Munich", calls.get(0).arguments().get("city"));

            Message reply2 =
                    drive(
                            model,
                            ChatRequest.builder()
                                    .messages(
                                            user,
                                            AiMessage.from(
                                                    ToolExecutionRequest.builder()
                                                            .id("call_0")
                                                            .name("get_weather")
                                                            .arguments("{\"city\":\"Munich\"}")
                                                            .build()),
                                            ToolExecutionResultMessage.from(
                                                    "call_0", "get_weather", "18C, sunny"))
                                    .parameters(withTool)
                                    .build());
            String text = reply2.text().strip();
            Object parsed = JsonCodec.parse(text);
            assertTrue(parsed instanceof Map, "schema-shaped answer expected: " + text);
            Map<?, ?> map = (Map<?, ?>) parsed;
            assertTrue(String.valueOf(map.get("city")).contains("Munich"), "grounded: " + text);
            assertTrue(map.get("temperature_c") instanceof Number, "numeric: " + text);
            assertEquals(18.0, ((Number) map.get("temperature_c")).doubleValue(), 0.01, text);
        }
    }
}
