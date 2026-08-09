package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.Tokenizer;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * PROTOTYPE of tools + JSON-schema response format as ONE constrained decode: think stays free,
 * calls stay the family's own syntax (free span between the call markers), and visible CONTENT can
 * only be schema JSON. The dispatch point - schema-cursor union call-marker union end-of-turn - is
 * hand-rolled here because the reply language does not (yet) allow a region to OPEN on a grammar
 * payload ("a region must open with a mark or a free hole"); a green run is the evidence that
 * gbnf-opening regions are the one missing engine piece.
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

    record Reply(List<Part.ToolCall> calls, String content) {}

    /** One request, one constrained decode: the tools+schema selection, hand-dispatched. */
    static Reply drive(JinferChatModel model, ChatRequest request) {
        ChatEngine.Prepared p = model.prepare(request);
        Tokenizer tokenizer = model.engine.loaded().tokenizer();
        int thinkOpen = SpecialTokens.find(tokenizer, "<think>").orElseThrow();
        int thinkClose = SpecialTokens.find(tokenizer, "</think>").orElseThrow();
        int callOpen = SpecialTokens.find(tokenizer, "<|tool_call_start|>").orElseThrow();
        int callClose = SpecialTokens.find(tokenizer, "<|tool_call_end|>").orElseThrow();
        int imEnd = SpecialTokens.find(tokenizer, "<|im_end|>").orElseThrow();
        int stop = model.engine.loaded().stopTokens().iterator().next();
        Grammar.Cursor content = Grammar.fromSchema(SCHEMA, tokenizer).cursor();
        long[] opening = content.admissible(); // the schema's first tokens, for dispatch

        boolean seededThink = false;
        for (int t : p.parserSeed()) seededThink |= t == thinkOpen;
        final int THINK = 0, DISPATCH = 1, CONTENT = 2, CALL = 3;
        int[] mode = {seededThink ? THINK : DISPATCH};
        List<Integer> callIds = new ArrayList<>();
        List<Integer> contentIds = new ArrayList<>();
        List<Part.ToolCall> calls = new ArrayList<>();
        Sampler argmax = com.qxotic.jinfer.FloatTensor::argmax;
        Sampler masked =
                logits -> {
                    switch (mode[0]) {
                        case THINK -> {
                            int t = argmax.sampleToken(logits); // reasoning samples free
                            if (t == thinkClose) mode[0] = DISPATCH;
                            return t;
                        }
                        case DISPATCH -> {
                            // the missing engine piece, hand-rolled: a plain token the schema
                            // admits opens constrained content; the call marker opens a call;
                            // the terminator ends the turn - the union IS the mask
                            int n = Math.toIntExact(logits.size());
                            for (int i = 0; i < n; i++) {
                                boolean inSchema = (opening[i >> 6] >>> (i & 63) & 1L) != 0;
                                if (!inSchema && i != callOpen && i != imEnd) {
                                    logits.setFloat(i, Float.NEGATIVE_INFINITY);
                                }
                            }
                            int t = argmax.sampleToken(logits);
                            if (t == callOpen) {
                                mode[0] = CALL;
                            } else if (t != imEnd) {
                                mode[0] = CONTENT;
                                content.tryAdvance(t);
                                contentIds.add(t);
                            }
                            return t;
                        }
                        case CONTENT -> {
                            if (!content.maskLogits(logits)) return stop; // complete: end turn
                            int t = argmax.sampleToken(logits);
                            // an ACCEPTING grammar state admits specials (empty bytes) so the
                            // model can end the turn - control tokens are never content text
                            if (SpecialTokens.isSpecial(tokenizer, t)) return t;
                            content.tryAdvance(t);
                            contentIds.add(t);
                            return t;
                        }
                        default -> {
                            int t = argmax.sampleToken(logits); // the call span is the model's
                            if (t == callClose) {
                                calls.addAll(ToolCallSyntax.parseBlock(decode(tokenizer, callIds)));
                                callIds.clear();
                                mode[0] = DISPATCH;
                            } else {
                                callIds.add(t);
                            }
                            return t;
                        }
                    }
                };
        model.engine.generate(p.encoded().prompt(), masked, 512, 0, token -> true);
        return new Reply(calls, decode(tokenizer, contentIds));
    }

    static String decode(Tokenizer tokenizer, List<Integer> ids) {
        int[] raw = ids.stream().mapToInt(Integer::intValue).toArray();
        return new String(tokenizer.decodeBytes(raw), StandardCharsets.UTF_8);
    }

    @Test
    void oneConstrainedDecodeServesBothToolCallsAndSchemaAnswers() {
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
            ChatRequest round1 =
                    ChatRequest.builder()
                            .messages(user)
                            .parameters(
                                    ChatRequestParameters.builder()
                                            .toolSpecifications(weather)
                                            .build())
                            .build();
            Reply reply1 = drive(model, round1);
            assertEquals(1, reply1.calls().size(), "the schema mask must not trap the call");
            assertEquals("get_weather", reply1.calls().get(0).name());
            assertEquals("Munich", reply1.calls().get(0).arguments().get("city"));

            ChatRequest round2 =
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
                            .parameters(
                                    ChatRequestParameters.builder()
                                            .toolSpecifications(weather)
                                            .build())
                            .build();
            Reply reply2 = drive(model, round2);
            String text = reply2.content().strip();
            Object parsed = JsonCodec.parse(text);
            assertTrue(parsed instanceof Map, "schema-shaped answer expected: " + text);
            Map<?, ?> map = (Map<?, ?>) parsed;
            assertTrue(String.valueOf(map.get("city")).contains("Munich"), "grounded: " + text);
            assertTrue(map.get("temperature_c") instanceof Number, "numeric: " + text);
            assertEquals(18.0, ((Number) map.get("temperature_c")).doubleValue(), 0.01, text);
        }
    }
}
