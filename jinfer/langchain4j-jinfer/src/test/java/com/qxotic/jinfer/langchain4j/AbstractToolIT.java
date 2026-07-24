package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The model-independent tool-calling contract, end-to-end against a real GGUF: complex declarations
 * (enums, integers, booleans, arrays, nested objects, no-parameter tools), argument wire fidelity
 * (numbers, unicode, quotes), multi-turn call history re-encoding, {@code REQUIRED} / {@code NONE}
 * semantics, and streaming atomicity (calls announced whole, payloads never streamed).
 *
 * <p>Parameterized by model: each concrete subclass names one GGUF (overridable via a system
 * property) and assume-skips when the file is absent. WHETHER the model chooses to call is an
 * assumption (model behavior); everything after a call happens - names, argument structure, echo
 * round-trips, forced/none guarantees - is an assertion (wire contract).
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class AbstractToolIT {

    /** The GGUF this subclass runs against. */
    abstract Path modelPath();

    static final ToolSpecification WEATHER =
            ToolSpecification.builder()
                    .name("get_weather")
                    .description("Get current weather for a city")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("city", "The city name")
                                    .addEnumProperty(
                                            "unit",
                                            List.of("celsius", "fahrenheit"),
                                            "Temperature unit")
                                    .required("city")
                                    .build())
                    .build();

    static final ToolSpecification TIME =
            ToolSpecification.builder()
                    .name("get_time")
                    .description("Get the current local time for a city")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("city")
                                    .required("city")
                                    .build())
                    .build();

    static final ToolSpecification FLIGHT =
            ToolSpecification.builder()
                    .name("book_flight")
                    .description("Book a flight")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("origin", "Departure city")
                                    .addStringProperty("destination", "Arrival city")
                                    .addIntegerProperty("passengers", "Number of passengers")
                                    .addEnumProperty(
                                            "cabin",
                                            List.of("economy", "business", "first"),
                                            "Cabin class")
                                    .addBooleanProperty("flexible", "Whether dates are flexible")
                                    .required("origin", "destination", "passengers")
                                    .build())
                    .build();

    static final ToolSpecification HOTELS =
            ToolSpecification.builder()
                    .name("search_hotels")
                    .description("Search hotels in a city")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("city")
                                    .addProperty(
                                            "filters",
                                            JsonObjectSchema.builder()
                                                    .description("Search filters")
                                                    .addProperty(
                                                            "stars",
                                                            JsonIntegerSchema.builder()
                                                                    .description(
                                                                            "Minimum star rating")
                                                                    .build())
                                                    .addProperty(
                                                            "amenities",
                                                            JsonArraySchema.builder()
                                                                    .items(
                                                                            JsonStringSchema
                                                                                    .builder()
                                                                                    .build())
                                                                    .description(
                                                                            "Required amenities")
                                                                    .build())
                                                    .build())
                                    .addStringProperty("checkin", "Check-in date, YYYY-MM-DD")
                                    .required("city")
                                    .build())
                    .build();

    static final ToolSpecification SEND =
            ToolSpecification.builder()
                    .name("send_message")
                    .description("Send a text message to the family group chat")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("text", "The exact message text to send")
                                    .required("text")
                                    .build())
                    .build();

    static final ToolSpecification REFRESH =
            ToolSpecification.builder()
                    .name("refresh_cache")
                    .description("Refresh the server-side cache")
                    .build();

    JinferChatModel model;

    @BeforeAll
    void load() {
        Assumptions.assumeTrue(Files.exists(modelPath()), "model not found: " + modelPath());
        model =
                JinferChatModel.builder()
                        .modelPath(modelPath())
                        .contextLength(4096)
                        .maxOutputTokens(1024)
                        .build();
    }

    // ---- helpers ----

    ChatResponse chat(List<ChatMessage> messages, ToolSpecification... tools) {
        return model.chat(
                ChatRequest.builder().messages(messages).toolSpecifications(tools).build());
    }

    ChatResponse ask(String user, ToolSpecification... tools) {
        return chat(List.of(UserMessage.from(user)), tools);
    }

    /** Assume-skips (model behavior) unless the response carries a call, then returns the first. */
    ToolExecutionRequest assumeCall(ChatResponse r) {
        Assumptions.assumeTrue(
                r.aiMessage().hasToolExecutionRequests(),
                "model chose not to call a tool: " + r.aiMessage().text());
        assertEquals(FinishReason.TOOL_EXECUTION, r.finishReason());
        return r.aiMessage().toolExecutionRequests().get(0);
    }

    /** The call's arguments as a parsed JSON object - the wire-fidelity view. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> args(ToolExecutionRequest call) {
        Object parsed = JsonCodec.parse(call.arguments());
        assertTrue(parsed instanceof Map, "arguments must be a JSON object: " + call.arguments());
        return (Map<String, Object>) parsed;
    }

    /** Numbers may arrive as JSON numbers or quoted strings; both must carry the value. */
    static long asLong(Object v) {
        if (v instanceof Number n) return n.longValue();
        return Long.parseLong(String.valueOf(v).trim());
    }

    record Streamed(
            ChatResponse response, String text, String thinking, List<CompleteToolCall> calls) {}

    Streamed stream(ChatRequest request) throws Exception {
        var done = new CompletableFuture<ChatResponse>();
        StringBuilder text = new StringBuilder();
        StringBuilder thinking = new StringBuilder();
        List<CompleteToolCall> calls = new ArrayList<>();
        model.streaming()
                .chat(
                        request,
                        new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialResponse(String partial) {
                                text.append(partial);
                            }

                            @Override
                            public void onPartialThinking(
                                    dev.langchain4j.model.chat.response.PartialThinking partial) {
                                thinking.append(partial.text());
                            }

                            @Override
                            public void onCompleteToolCall(CompleteToolCall call) {
                                calls.add(call);
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
        ChatResponse r = done.get(5, TimeUnit.MINUTES);
        return new Streamed(r, text.toString(), thinking.toString(), calls);
    }

    /** No family's raw call syntax may surface as content. */
    static void assertNoCallSyntax(String content) {
        assertTrue(
                !content.contains("get_weather(")
                        && !content.contains("\"city\"")
                        && !content.contains("city=")
                        && !content.contains("city:"),
                "call payload leaked into content: " + content);
    }

    // ---- the contract ----

    @Test
    void toolRoundTrip() {
        ChatResponse first = ask("What is the weather in Paris? Use the tool.", WEATHER);
        ToolExecutionRequest call = assumeCall(first);
        assertEquals("get_weather", call.name());
        assertTrue(String.valueOf(args(call).get("city")).contains("Paris"), call.arguments());

        ChatResponse second =
                chat(
                        List.of(
                                UserMessage.from("What is the weather in Paris? Use the tool."),
                                first.aiMessage(),
                                ToolExecutionResultMessage.from(
                                        call.id(), call.name(), "18C, sunny")),
                        WEATHER);
        assertNotNull(second.aiMessage().text());
        assertTrue(second.aiMessage().text().contains("18"), second.aiMessage().text());
    }

    @Test
    void picksTheRightToolAmongMany() {
        ChatResponse r =
                ask(
                        "What is the weather in Tokyo? Use the right tool.",
                        WEATHER,
                        TIME,
                        FLIGHT,
                        REFRESH);
        ToolExecutionRequest call = assumeCall(r);
        assertEquals("get_weather", call.name());
        assertTrue(String.valueOf(args(call).get("city")).contains("Tokyo"), call.arguments());
    }

    @Test
    void multiArgumentCall() {
        ChatResponse r =
                ask(
                        "Book a flight from Zurich to Lisbon for 3 passengers in economy. Use the"
                                + " tool.",
                        FLIGHT);
        ToolExecutionRequest call = assumeCall(r);
        assertEquals("book_flight", call.name());
        Map<String, Object> a = args(call);
        assertTrue(String.valueOf(a.get("origin")).contains("Zurich"), call.arguments());
        assertTrue(String.valueOf(a.get("destination")).contains("Lisbon"), call.arguments());
        assertEquals(3, asLong(a.get("passengers")), call.arguments());
    }

    @Test
    void nestedObjectArguments() {
        ChatResponse r =
                ask(
                        "Search hotels in Rome: at least 4 stars, must have a pool, check-in"
                                + " 2026-08-01. Use the tool.",
                        HOTELS);
        ToolExecutionRequest call = assumeCall(r);
        assertEquals("search_hotels", call.name());
        Map<String, Object> a = args(call);
        assertTrue(String.valueOf(a.get("city")).contains("Rome"), call.arguments());
        // whether the model uses the nested filters is its choice; when it does, the structure
        // must round-trip as REAL JSON structure, not stringified fragments
        if (a.get("filters") instanceof Map<?, ?> filters) {
            if (filters.containsKey("stars")) {
                assertEquals(4, asLong(filters.get("stars")), call.arguments());
            }
            if (filters.get("amenities") instanceof List<?> amenities) {
                assertTrue(!amenities.isEmpty(), call.arguments());
            }
        }
    }

    @Test
    void unicodeAndQuotesInArguments() {
        ChatResponse r =
                ask(
                        "Send exactly this message to the group chat, verbatim: He said \"grüß"
                                + " dich\" 🌊. Use the tool.",
                        SEND);
        ToolExecutionRequest call = assumeCall(r);
        assertEquals("send_message", call.name());
        String text = String.valueOf(args(call).get("text"));
        assertTrue(text.contains("grüß"), call.arguments());
        assertTrue(text.contains("🌊"), call.arguments());
        assertTrue(text.contains("\"") || text.contains("“"), call.arguments());
    }

    @Test
    void noParameterTool() {
        ChatResponse r = ask("Please refresh the cache now using the tool.", REFRESH);
        ToolExecutionRequest call = assumeCall(r);
        assertEquals("refresh_cache", call.name());
        String raw = call.arguments() == null ? "" : call.arguments().strip();
        assertTrue(
                raw.isEmpty() || raw.equals("{}") || args(call).isEmpty(),
                "no-parameter call must carry no arguments: " + raw);
    }

    @Test
    void jsonToolResult() {
        ChatResponse first = ask("What is the weather in Paris? Use the tool.", WEATHER);
        ToolExecutionRequest call = assumeCall(first);
        ChatResponse second =
                chat(
                        List.of(
                                UserMessage.from("What is the weather in Paris? Use the tool."),
                                first.aiMessage(),
                                ToolExecutionResultMessage.from(
                                        call.id(),
                                        call.name(),
                                        "{\"temp_c\": 18, \"conditions\": \"sunny\","
                                                + " \"humidity\": 0.62}")),
                        WEATHER);
        assertTrue(second.aiMessage().text().contains("18"), second.aiMessage().text());
    }

    /**
     * Drives the agentic loop on {@code history}: while the model calls tools, results are appended
     * by name from {@code results} and the model re-asked. Returns the final text response; appends
     * everything it adds to {@code history}.
     */
    ChatResponse converse(
            List<ChatMessage> history, Map<String, String> results, ToolSpecification... tools) {
        for (int round = 0; round < 4; round++) {
            ChatResponse r = chat(history, tools);
            if (!r.aiMessage().hasToolExecutionRequests()) return r;
            history.add(r.aiMessage());
            for (ToolExecutionRequest call : r.aiMessage().toolExecutionRequests()) {
                String result = results.get(call.name());
                assertNotNull(result, "model called an unoffered tool: " + call.name());
                history.add(ToolExecutionResultMessage.from(call.id(), call.name(), result));
            }
        }
        throw new AssertionError("model never stopped calling tools: " + history);
    }

    @Test
    void multiTurnToolLoop() {
        Map<String, String> results = Map.of("get_weather", "18C, sunny", "get_time", "14:32");
        // round trip 1: weather
        List<ChatMessage> history = new ArrayList<>();
        history.add(UserMessage.from("What is the weather in Paris? Use the tools."));
        ChatResponse first = chat(history, WEATHER, TIME);
        ToolExecutionRequest weather = assumeCall(first);
        assertEquals("get_weather", weather.name());
        history.add(first.aiMessage());
        history.add(ToolExecutionResultMessage.from(weather.id(), weather.name(), "18C, sunny"));
        ChatResponse answer1 = converse(history, results, WEATHER, TIME);
        assertTrue(answer1.aiMessage().text().contains("18"), answer1.aiMessage().text());
        history.add(answer1.aiMessage());

        // round trip 2 on the SAME history: the call turns above re-encode as history. The wire
        // contract here is that the multi-round history frames and parses cleanly; whether the
        // model USES the second result is capability (small models hallucinate results inline
        // and ignore the fed one - observed on gemma E2B), so that part is assumption-gated.
        history.add(UserMessage.from("And what time is it there right now? Use the tools."));
        ChatResponse second = chat(history, WEATHER, TIME);
        ToolExecutionRequest time = assumeCall(second);
        assertEquals("get_time", time.name());
        history.add(second.aiMessage());
        history.add(ToolExecutionResultMessage.from(time.id(), time.name(), "14:32"));
        ChatResponse answer2 = converse(history, results, WEATHER, TIME);
        String text2 = answer2.aiMessage().text();
        Assumptions.assumeTrue(
                text2 != null && (text2.contains("14:32") || text2.contains("2:32")),
                "model did not use the second tool result: " + text2);
    }

    @Test
    void requiredForcesAnOfferedTool() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(UserMessage.from("Say hello."))
                                .toolSpecifications(WEATHER, TIME)
                                .toolChoice(ToolChoice.REQUIRED)
                                .build());
        assertTrue(
                r.aiMessage().hasToolExecutionRequests(),
                "REQUIRED must force a call: " + r.aiMessage());
        assertEquals(FinishReason.TOOL_EXECUTION, r.finishReason());
        String name = r.aiMessage().toolExecutionRequests().get(0).name();
        assertTrue(
                Set.of("get_weather", "get_time").contains(name),
                "forced call must name an offered tool: " + name);
    }

    @Test
    void noneNeverCalls() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the tool."))
                                .toolSpecifications(WEATHER)
                                .toolChoice(ToolChoice.NONE)
                                .build());
        assertTrue(
                !r.aiMessage().hasToolExecutionRequests(),
                "NONE must prevent tool calls: " + r.aiMessage());
    }

    @Test
    void streamingToolRoundTrip() throws Exception {
        ChatRequest ask =
                ChatRequest.builder()
                        .messages(UserMessage.from("What is the weather in Paris? Use the tool."))
                        .toolSpecifications(WEATHER)
                        .build();
        Streamed first = stream(ask);
        Assumptions.assumeTrue(
                first.response().aiMessage().hasToolExecutionRequests(),
                "model chose not to call the tool: " + first.response().aiMessage().text());
        assertEquals(FinishReason.TOOL_EXECUTION, first.response().finishReason());
        List<ToolExecutionRequest> requests = first.response().aiMessage().toolExecutionRequests();
        var call = requests.get(0);
        assertEquals("get_weather", call.name());
        // every call was announced whole before the response, and no payload ever streamed
        assertEquals(requests.size(), first.calls().size());
        for (int i = 0; i < requests.size(); i++) {
            assertEquals(requests.get(i), first.calls().get(i).toolExecutionRequest());
        }
        assertNoCallSyntax(first.text());

        // answer EVERY announced call (a model may emit more than one per turn)
        List<ChatMessage> followUp = new ArrayList<>();
        followUp.add(UserMessage.from("What is the weather in Paris? Use the tool."));
        followUp.add(first.response().aiMessage());
        for (ToolExecutionRequest r : requests) {
            followUp.add(ToolExecutionResultMessage.from(r.id(), r.name(), "18C, sunny"));
        }
        Streamed second =
                stream(
                        ChatRequest.builder()
                                .messages(followUp)
                                .toolSpecifications(WEATHER)
                                .build());
        // the streamed fragments and the final message agree
        assertEquals(second.response().aiMessage().text(), second.text());
        assertTrue(second.text().contains("18"), second.text());
    }

    @Test
    void streamingRequiredForcesAnOfferedTool() throws Exception {
        Streamed r =
                stream(
                        ChatRequest.builder()
                                .messages(UserMessage.from("Say hello."))
                                .toolSpecifications(WEATHER, TIME)
                                .toolChoice(ToolChoice.REQUIRED)
                                .build());
        assertTrue(
                r.response().aiMessage().hasToolExecutionRequests(),
                "REQUIRED must force a call: " + r.response().aiMessage());
        assertEquals(FinishReason.TOOL_EXECUTION, r.response().finishReason());
        assertEquals(r.response().aiMessage().toolExecutionRequests().size(), r.calls().size());
        String name = r.calls().get(0).toolExecutionRequest().name();
        assertTrue(
                Set.of("get_weather", "get_time").contains(name),
                "forced call must name an offered tool: " + name);
        // isBlank, not isEmpty: whitespace around a family's call block is framing, not content
        assertTrue(r.text().isBlank(), "a forced call streams no content: '" + r.text() + "'");
    }
}
