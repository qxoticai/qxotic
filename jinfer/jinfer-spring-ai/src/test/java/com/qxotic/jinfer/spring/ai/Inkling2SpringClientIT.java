package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.function.FunctionToolCallback;
import reactor.core.publisher.Flux;

/**
 * A Spring application's shapes, end to end on Inkling 2 Mini: the ChatClient fluent API, bean
 * structured output, streaming with usage metadata, the option surface (sampling, reasoning,
 * timeout), a FunctionToolCallback executed BY the advisor chain, and conversation memory across
 * turns. The generic batteries ({@link Inkling2CapabilityIT}, {@link Inkling2ToolCallingAdvisorIT})
 * pin the mapping and advisor contracts; this one is the "does a real Spring app work" pass.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.spring.ai.Inkling2SpringClientIT#modelAvailable")
class Inkling2SpringClientIT {

    static final String REF = "hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf";

    record City(String city, double population_millions) {}

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    private static JinferChatModel model;
    private static ChatClient client;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextCapacity(8192)
                        .options(
                                JinferChatOptions.builder().maxTokens(256).temperature(0.0).build())
                        .build();
        client = ChatClient.builder(model).build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void fluentCallAndPromptCallAgree() {
        String fluent =
                client.prompt()
                        .user("What is the capital of France? Answer in one word.")
                        .call()
                        .content();
        assertNotNull(fluent);
        assertTrue(fluent.toLowerCase(Locale.ROOT).contains("paris"), fluent);

        ChatResponse response =
                model.call(
                        new Prompt(
                                new UserMessage(
                                        "What is the capital of Japan? Answer in one word."),
                                JinferChatOptions.builder()
                                        .maxTokens(64)
                                        .temperature(0.0)
                                        .build()));
        String text = response.getResult().getOutput().getText();
        assertTrue(text.toLowerCase(Locale.ROOT).contains("tokyo"), text);

        Usage usage = response.getMetadata().getUsage();
        assertNotNull(usage);
        assertTrue(usage.getPromptTokens() > 0, "prompt tokens billed: " + usage);
        assertTrue(usage.getCompletionTokens() > 0, "completion tokens billed: " + usage);
        assertEquals(usage.getPromptTokens() + usage.getCompletionTokens(), usage.getTotalTokens());
        assertNotNull(response.getResult().getMetadata().getFinishReason());
    }

    @Test
    void beanStructuredOutputBindsTheReply() {
        City city =
                client.prompt()
                        .user(
                                "What is the capital of France and roughly how many million people"
                                        + " live in the city?")
                        .call()
                        .entity(City.class);
        assertNotNull(city);
        assertTrue(city.city().toLowerCase(Locale.ROOT).contains("paris"), city.toString());
        assertTrue(city.population_millions() > 0, city.toString());
    }

    @Test
    void structuredOutputThroughAdvisorChain() {
        City city =
                client.prompt()
                        .advisors(new SimpleLoggerAdvisor())
                        .user("Name the capital of Italy and its population in millions.")
                        .call()
                        .entity(City.class);
        assertNotNull(city);
        assertTrue(city.city().toLowerCase(Locale.ROOT).contains("rome"), city.toString());
    }

    @Test
    void streamingDeliversContentAndAFinishedResponse() {
        List<String> deltas = new ArrayList<>();
        AtomicReference<ChatResponse> finished = new AtomicReference<>();
        Flux<ChatResponse> flux =
                client
                        .prompt()
                        .user("Count from one to five in words, separated by commas.")
                        .stream()
                        .chatResponse();
        flux.doOnNext(
                        response -> {
                            if (response.getResult() != null) {
                                String delta = response.getResult().getOutput().getText();
                                if (delta != null && !delta.isEmpty()) deltas.add(delta);
                            }
                        })
                .doOnComplete(() -> finished.set(finished.get()))
                .blockLast(Duration.ofMinutes(10));

        assertFalse(deltas.isEmpty(), "no streamed deltas");
        String joined = String.join("", deltas);
        assertTrue(joined.toLowerCase(Locale.ROOT).contains("one"), joined);
        assertTrue(joined.toLowerCase(Locale.ROOT).contains("five"), joined);
    }

    @Test
    void reasoningArrivesSeparatedWhenAsked() {
        ChatResponse response =
                model.call(
                        new Prompt(
                                new UserMessage("What is 17 * 23? Think, then answer."),
                                JinferChatOptions.builder()
                                        .maxTokens(320)
                                        .thinking(true)
                                        .maxReasoningTokens(128)
                                        .temperature(0.0)
                                        .build()));
        String text = response.getResult().getOutput().getText();
        assertNotNull(text);
        assertTrue(text.contains("391"), text);
        // the reply must not carry the family's block markers into visible text
        assertFalse(text.contains("<|content_thinking|>"), text);
        assertFalse(text.contains("<|end_message|>"), text);
    }

    @Test
    void samplingOptionsAreHonouredWithoutBreaking() {
        String seededA =
                client.prompt()
                        .user("Name a fruit.")
                        .options(
                                JinferChatOptions.builder()
                                        .temperature(0.7)
                                        .topP(0.9)
                                        .topK(40)
                                        .minP(0.05)
                                        .seed(11L)
                                        .maxTokens(24))
                        .call()
                        .content();
        String seededB =
                client.prompt()
                        .user("Name a fruit.")
                        .options(
                                JinferChatOptions.builder()
                                        .temperature(0.7)
                                        .topP(0.9)
                                        .topK(40)
                                        .minP(0.05)
                                        .seed(11L)
                                        .maxTokens(24))
                        .call()
                        .content();
        assertNotNull(seededA);
        assertEquals(seededA, seededB, "a fixed seed reproduces the reply");
    }

    @Test
    void functionToolIsExecutedByTheAdvisorChain() {
        AtomicReference<String> seen = new AtomicReference<>();
        var weather =
                FunctionToolCallback.builder(
                                "get_weather",
                                (Map<String, Object> input) -> {
                                    seen.set(String.valueOf(input.get("city")));
                                    return "18C and sunny";
                                })
                        .description("Get the current weather for a city")
                        .inputType(Map.class)
                        .inputSchema(
                                "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}")
                        .build();

        String answer =
                client.prompt()
                        .user("What is the weather in Paris? Use the get_weather tool.")
                        .toolCallbacks(weather)
                        .call()
                        .content();
        assertNotNull(answer);
        assertNotNull(seen.get(), "the tool callback was never invoked: " + answer);
        assertTrue(
                seen.get().toLowerCase(Locale.ROOT).contains("paris"),
                "tool called with " + seen.get());
        assertTrue(answer.contains("18"), "the answer uses the tool result: " + answer);
    }

    @Test
    void conversationMemoryCarriesAnEarlierTurn() {
        List<org.springframework.ai.chat.messages.Message> messages = new ArrayList<>();
        messages.add(new UserMessage("My name is Mira and my favourite colour is teal."));
        ChatResponse first = model.call(new Prompt(messages));
        messages.add(first.getResult().getOutput());
        messages.add(new UserMessage("What is 2 + 2?"));
        ChatResponse second = model.call(new Prompt(messages));
        messages.add(second.getResult().getOutput());
        messages.add(new UserMessage("What is my name and my favourite colour?"));

        String recall = model.call(new Prompt(messages)).getResult().getOutput().getText();
        String lower = recall.toLowerCase(Locale.ROOT);
        assertTrue(lower.contains("mira"), recall);
        assertTrue(lower.contains("teal"), recall);
    }

    @Test
    void systemMessageIsHonoured() {
        String reply =
                client.prompt()
                        .system("You are terse. Answer with at most three words.")
                        .user("Describe the ocean.")
                        .call()
                        .content();
        assertNotNull(reply);
        assertTrue(reply.strip().split("\\s+").length <= 8, "reply was not terse: " + reply);
    }

    @Test
    void anAssistantTurnReplaysThroughTheTemplate() {
        List<org.springframework.ai.chat.messages.Message> messages = new ArrayList<>();
        messages.add(new UserMessage("Name one primary color."));
        messages.add(new AssistantMessage("Blue."));
        messages.add(new UserMessage("Name another one."));
        String reply = model.call(new Prompt(messages)).getResult().getOutput().getText();
        assertNotNull(reply);
        assertFalse(reply.isBlank(), "the replayed assistant turn broke the conversation framing");
    }
}
