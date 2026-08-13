package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The grammar/seed laws E2E on the smallest chat model (LFM2.5 350M): a GBNF gate admits only its
 * language; temperature-0 requests replay byte-identically and different seeds diverge (byte
 * identity at temperature &gt; 0 is deliberately NOT asserted: jam's run-to-run FP jitter flips
 * near-tie samples); a standing grammar on the MODEL constrains plain requests (the framework-merge
 * path AiServices rides); the two conflict rejections are loud.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class JinferRequestParametersIT {

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"))
                        .contextLength(2048)
                        .maxOutputTokens(48)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void grammarAdmitsOnlyItsLanguage() {
        ChatRequest gated =
                ChatRequest.builder()
                        .messages(UserMessage.from("Is the sky blue? Answer strictly yes or no."))
                        .parameters(
                                JinferChatRequestParameters.builder()
                                        .grammar("root ::= \"yes\" | \"no\"")
                                        .build())
                        .build();
        String text = model.chat(gated).aiMessage().text();
        assertTrue(text.equals("yes") || text.equals("no"), "constrained output: '" + text + "'");
    }

    @Test
    void replayIsByteIdenticalAtTemperatureZero() {
        // replay identity, pinned where the backend can honor it: temperature 0, hot-vs-hot
        // (cold-vs-warm tier drift reaches whole logits and flips argmax; at temperature 1.0
        // even warm jitter flips near-ties - seed liveness is differentSeedsDiverge's job,
        // divergence being drift-immune)
        ChatRequest req =
                ChatRequest.builder()
                        .messages(UserMessage.from("Invent a name for a sailing boat."))
                        .parameters(
                                JinferChatRequestParameters.builder()
                                        .temperature(0.0)
                                        .maxOutputTokens(24)
                                        .build())
                        .build();
        for (int i = 0; i < 4; i++) model.chat(req); // hot-vs-hot, like every numerics gate
        String a = model.chat(req).aiMessage().text();
        String b = model.chat(req).aiMessage().text();
        assertEquals(a, b, "temperature 0 must replay identically");
    }

    @Test
    void differentSeedsDiverge() {
        String a = model.chat(creative(1L)).aiMessage().text();
        String b = model.chat(creative(2L)).aiMessage().text();
        assertNotEquals(a, b, "different seeds at temperature 1.0 should diverge");
    }

    @Test
    void standingGrammarConstrainsPlainRequests() {
        // the AiServices path: jinfer defaults on the model, a request with NO jinfer parameters -
        // the framework merge (defaults.overrideWith(request)) must carry the grammar through
        try (JinferChatModel gate =
                JinferChatModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"))
                        .contextLength(2048)
                        .defaultRequestParameters(
                                JinferChatRequestParameters.builder()
                                        .grammar("root ::= \"positive\" | \"negative\"")
                                        .maxOutputTokens(8)
                                        .build())
                        .build()) {
            String text =
                    gate.chat(UserMessage.from("Sentiment of: 'this update is wonderful'"))
                            .aiMessage()
                            .text();
            assertTrue(
                    text.equals("positive") || text.equals("negative"),
                    "standing grammar must constrain plain requests: '" + text + "'");
        }
    }

    @Test
    void grammarConflictsAreLoud() {
        var grammarParams = JinferChatRequestParameters.builder().grammar("root ::= \"x\"").build();
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        model.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("hi"))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar("root ::= \"x\"")
                                                        .toolSpecifications(
                                                                ToolSpecification.builder()
                                                                        .name("noop")
                                                                        .build())
                                                        .build())
                                        .build()),
                "grammar + tools");
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        model.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("hi"))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar("root ::= \"x\"")
                                                        .responseFormat(ResponseFormat.JSON)
                                                        .build())
                                        .build()),
                "grammar + JSON response format");
        // and the params object alone is inert - conflicts reject at chat time, loudly
        assertEquals("root ::= \"x\"", grammarParams.grammar());
    }

    @Test
    void classificationIntoAClosedLabelSet() {
        String labels = "root ::= \"billing\" | \"shipping\" | \"warranty\" | \"other\"";
        String text =
                model.chat(
                                ChatRequest.builder()
                                        .messages(
                                                UserMessage.from(
                                                        "Classify this support ticket: 'my parcel"
                                                                + " never arrived'. Answer with the"
                                                                + " category only."))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar(labels)
                                                        .build())
                                        .build())
                        .aiMessage()
                        .text();
        assertTrue(
                List.of("billing", "shipping", "warranty", "other").contains(text),
                "label: '" + text + "'");
    }

    @Test
    void flatJsonObjectGrammarForbidsNesting() {
        // a SHALLOW object by construction: string values cannot contain quotes, braces or
        // brackets, so no nested {} or [] can ever appear - the shape is proven, not prompted
        String flatJson =
                "root ::= \"{\" ws \"\\\"label\\\"\" ws \":\" ws str ws \",\" ws"
                        + " \"\\\"confidence\\\"\" ws \":\" ws num ws \"}\"\n"
                        + "str ::= \"\\\"\" [a-zA-Z0-9 _-]* \"\\\"\"\n"
                        + "num ::= [0-9] | [0-9] \".\" [0-9] [0-9]?\n"
                        + "ws ::= \" \"?";
        String text =
                model.chat(
                                ChatRequest.builder()
                                        .messages(
                                                UserMessage.from(
                                                        "Classify 'the app crashes on startup' and"
                                                            + " give your confidence from 0 to 1 as"
                                                            + " JSON with fields label and"
                                                            + " confidence."))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar(flatJson)
                                                        .build())
                                        .build())
                        .aiMessage()
                        .text();
        Object parsed = Json.parse(text); // grammar guarantees JSON
        assertTrue(parsed instanceof Map<?, ?>, text);
        Map<?, ?> map = (Map<?, ?>) parsed;
        assertTrue(map.get("label") instanceof String, "label: " + text);
        assertTrue(map.get("confidence") instanceof Number, "confidence: " + text);
        assertTrue(
                !text.contains("[") && text.indexOf('{', 1) < 0, "flat by construction: " + text);
    }

    @Test
    void digitsOnlyExtraction() {
        String text =
                model.chat(
                                ChatRequest.builder()
                                        .messages(
                                                UserMessage.from(
                                                        "How many legs does a spider have? Answer"
                                                                + " with the number only."))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar("root ::= [0-9] [0-9]?")
                                                        .build())
                                        .build())
                        .aiMessage()
                        .text();
        int n = Integer.parseInt(text); // cannot throw: digits by construction
        assertTrue(n > 0, "legs: " + n);
    }

    @Test
    void multiTokenGrammarComposedWithSeedReplays() {
        String semver =
                "root ::= level \" \" num \".\" num \".\" num\n"
                        + "level ::= \"major\" | \"minor\" | \"patch\"\n"
                        + "num ::= [0-9] [0-9]?";
        ChatRequest req =
                ChatRequest.builder()
                        .messages(
                                UserMessage.from(
                                        "Current version 2.14.3. A public API was removed. Propose"
                                                + " the version bump."))
                        .parameters(
                                JinferChatRequestParameters.builder()
                                        .grammar(semver)
                                        .seed(99L)
                                        .build())
                        .build();
        String a = model.chat(req).aiMessage().text();
        String b = model.chat(req).aiMessage().text();
        assertTrue(a.matches("(major|minor|patch) \\d{1,2}\\.\\d{1,2}\\.\\d{1,2}"), a);
        assertEquals(a, b, "grammar + seed must replay identically");
    }

    @Test
    void streamingHonorsTheGrammar() throws Exception {
        var streaming = model.streaming();
        var done = new CompletableFuture<ChatResponse>();
        StringBuilder partials = new StringBuilder();
        streaming.chat(
                ChatRequest.builder()
                        .messages(UserMessage.from("Is water wet? Answer strictly yes or no."))
                        .parameters(
                                JinferChatRequestParameters.builder()
                                        .grammar("root ::= \"yes\" | \"no\"")
                                        .build())
                        .build(),
                new StreamingChatResponseHandler() {
                    @Override
                    public void onPartialResponse(String partialResponse) {
                        partials.append(partialResponse);
                    }

                    @Override
                    public void onCompleteResponse(ChatResponse response) {
                        done.complete(response);
                    }

                    @Override
                    public void onError(Throwable error) {
                        done.completeExceptionally(error);
                    }
                });
        var response = done.get(60, TimeUnit.SECONDS);
        String text = response.aiMessage().text();
        assertTrue(text.equals("yes") || text.equals("no"), "streamed: '" + text + "'");
        assertEquals(text, partials.toString(), "partials must concatenate to the final text");
    }

    @Test
    void streamingMatchesBlockingAtTemperatureZero() throws Exception {
        // the shared-pipeline law, pinned where it is provable: temperature 0 AND hot-vs-hot.
        // Cold-vs-warm prefills differ by 0.2-8 LOGITS (JIT tier reassociation) - enough to
        // flip argmax - and at temperature 1.0 even the warm ~2e-6 jitter flips any near-tie
        // on the trajectory (observed: two identical blocking runs in one warm JVM answered
        // differently). Warm both drivers first, then compare argmax to argmax
        ChatRequest req =
                ChatRequest.builder()
                        .messages(UserMessage.from("Invent a name for a sailing boat."))
                        .parameters(
                                JinferChatRequestParameters.builder()
                                        .temperature(0.0)
                                        .maxOutputTokens(24)
                                        .build())
                        .build();
        for (int i = 0; i < 4; i++) model.chat(req); // hot-vs-hot: tier drift is 0.2-8 logits
        streamText(req); // and the streaming driver thread warms its own call path
        String blocking = model.chat(req).aiMessage().text();
        String streamed = streamText(req);
        assertEquals(blocking, streamed, "streaming and blocking share one generation path");
    }

    @Test
    void streamingSeedIsLive() throws Exception {
        // the seed reaches the streaming sampler: different seeds diverge (drift-immune - two
        // seeds colliding on the same 24-token reply would be astronomically unlucky)
        assertNotEquals(
                streamText(creative(1L)),
                streamText(creative(2L)),
                "different seeds at temperature 1.0 should diverge in streaming too");
    }

    private String streamText(ChatRequest request) throws Exception {
        var done = new CompletableFuture<ChatResponse>();
        model.streaming()
                .chat(
                        request,
                        new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialResponse(String partialResponse) {}

                            @Override
                            public void onCompleteResponse(ChatResponse response) {
                                done.complete(response);
                            }

                            @Override
                            public void onError(Throwable error) {
                                done.completeExceptionally(error);
                            }
                        });
        return done.get(60, TimeUnit.SECONDS).aiMessage().text();
    }

    private static ChatRequest creative(long seed) {
        return ChatRequest.builder()
                .messages(UserMessage.from("Invent a name for a sailing boat. One name only."))
                .parameters(
                        JinferChatRequestParameters.builder()
                                .seed(seed)
                                .temperature(1.0)
                                .maxOutputTokens(24)
                                .build())
                .build();
    }
}
