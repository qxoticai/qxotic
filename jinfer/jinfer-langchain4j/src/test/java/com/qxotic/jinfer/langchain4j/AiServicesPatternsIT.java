package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.Result;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import dev.langchain4j.service.tool.ToolExecution;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The AiServices idioms professional langchain4j code is actually written in, over one shared
 * engine: {@link TokenStream} streaming (plain and through the automatic tool loop),
 * {@code @MemoryId} per-user conversations, and {@code @SystemMessage}/{@code @UserMessage}
 * templates. Model-gated: assume-skips without the GGUF.
 */
@Tag("integration")
class AiServicesPatternsIT {

    static final Path MODEL = ModelFixture.LFM25_8B_Q8.path();

    static JinferChatModel model;
    static JinferStreamingChatModel streaming; // a view over the SAME engine: one GGUF load

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(256)
                        .thinking(false)
                        .seed(7L)
                        .build();
        streaming = model.streaming();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    interface StreamingAssistant {
        TokenStream chat(String message);
    }

    @Test
    void tokenStreamDeltasConcatenateToTheCompleteResponse() throws Exception {
        StreamingAssistant assistant =
                AiServices.builder(StreamingAssistant.class).streamingChatModel(streaming).build();

        StringBuilder partials = new StringBuilder();
        CompletableFuture<ChatResponse> done = new CompletableFuture<>();
        assistant
                .chat("Reply with a short one-sentence greeting.")
                .onPartialResponse(partials::append)
                .onCompleteResponse(done::complete)
                .onError(done::completeExceptionally)
                .start();

        ChatResponse response = done.get(120, TimeUnit.SECONDS);
        assertFalse(response.aiMessage().text().isBlank());
        assertEquals(
                response.aiMessage().text(),
                partials.toString(),
                "the streamed deltas must concatenate to exactly the final text");
    }

    static class ServerRoom {
        @Tool("Reads the server room's temperature sensor, in Celsius")
        double temperature() {
            return 41.5;
        }
    }

    interface Watchman {
        TokenStream ask(String question);
    }

    @Test
    void tokenStreamRunsTheToolLoopAndAnnouncesExecutions() throws Exception {
        Watchman watchman =
                AiServices.builder(Watchman.class)
                        .streamingChatModel(streaming)
                        .tools(new ServerRoom())
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .build();

        List<ToolExecution> executed = new CopyOnWriteArrayList<>();
        CompletableFuture<ChatResponse> done = new CompletableFuture<>();
        watchman.ask("What is the server room temperature right now? Answer in one sentence.")
                .onToolExecuted(executed::add)
                .onPartialResponse(delta -> {})
                .onCompleteResponse(done::complete)
                .onError(done::completeExceptionally)
                .start();

        ChatResponse response = done.get(240, TimeUnit.SECONDS);
        assertTrue(
                executed.stream().anyMatch(e -> e.request().name().equals("temperature")),
                "the tool loop never called the sensor");
        assertTrue(response.aiMessage().text().contains("41"), response.aiMessage().text());
    }

    interface Concierge {
        String chat(@MemoryId String user, @UserMessage String message);
    }

    @Test
    void memoryIdKeepsInterleavedUsersApart() {
        // two conversations alternating over ONE engine - the per-user memory must stay separate,
        // and the flip-flopping histories also exercise the session cache's prefix matching
        Concierge concierge =
                AiServices.builder(Concierge.class)
                        .chatModel(model)
                        .chatMemoryProvider(id -> MessageWindowChatMemory.withMaxMessages(10))
                        .build();

        concierge.chat("alice", "My name is Alice and my favorite color is blue. Reply only OK.");
        concierge.chat("bob", "My name is Bob and my favorite color is green. Reply only OK.");
        String alice = concierge.chat("alice", "What is my favorite color? Answer in one word.");
        String bob = concierge.chat("bob", "What is my favorite color? Answer in one word.");

        assertTrue(alice.toLowerCase().contains("blue"), "alice got: " + alice);
        assertTrue(bob.toLowerCase().contains("green"), "bob got: " + bob);
        assertFalse(alice.toLowerCase().contains("green"), "bob's memory leaked into alice's");
        assertFalse(bob.toLowerCase().contains("blue"), "alice's memory leaked into bob's");
    }

    interface Translator {
        @SystemMessage("You translate English to {{lang}}. Reply with ONLY the translation.")
        @UserMessage("Translate: {{text}}")
        String translate(@V("lang") String lang, @V("text") String text);
    }

    @Test
    void promptTemplatesRenderThroughTheAdapter() {
        Translator translator = AiServices.create(Translator.class, model);
        String german = translator.translate("German", "Good morning");
        assertTrue(german.toLowerCase().contains("guten morgen"), german);
    }

    interface Chat {
        String chat(String message);
    }

    @Test
    void tokenWindowMemoryEvictsByTheModelsOwnCounts() {
        // the stock ChatMemoryExamples pattern: TokenWindowChatMemory sized by the provider's
        // tokenCountEstimator - the model's REAL vocabulary, not a heuristic
        ChatMemory memory = TokenWindowChatMemory.withMaxTokens(80, model.tokenCountEstimator());
        Chat chat = AiServices.builder(Chat.class).chatModel(model).chatMemory(memory).build();

        chat.chat("The secret launch code is PINEAPPLE. Reply only OK.");
        chat.chat("Please describe the ocean in two full sentences.");
        chat.chat("Please describe a forest in two full sentences.");

        assertTrue(
                memory.messages().stream().noneMatch(m -> m.toString().contains("PINEAPPLE")),
                "an 80-token window must have evicted the first exchange: " + memory.messages());
        assertFalse(chat.chat("Say hello in one word.").isBlank(), "the service must keep working");
    }

    enum Sentiment {
        POSITIVE,
        NEUTRAL,
        NEGATIVE
    }

    interface SentimentAnalyzer {
        @UserMessage("Classify the sentiment of this review: {{it}}")
        Sentiment analyze(String review);
    }

    @Test
    void enumReturnClassifies() {
        SentimentAnalyzer analyzer = AiServices.create(SentimentAnalyzer.class, model);
        assertEquals(Sentiment.POSITIVE, analyzer.analyze("I love this product, it works great!"));
        assertEquals(Sentiment.NEGATIVE, analyzer.analyze("Broke on day one, complete waste."));
    }

    record Person(String name, int age) {}

    interface PeopleExtractor {
        @UserMessage("Extract every person mentioned in: {{it}}")
        List<Person> extractAll(String text);
    }

    @Test
    void listOfPojosExtracts() {
        List<Person> people =
                AiServices.create(PeopleExtractor.class, model)
                        .extractAll("Alice is 30 years old and her brother Bob is 25.");
        assertEquals(2, people.size(), "got: " + people);
        assertTrue(people.contains(new Person("Alice", 30)), "got: " + people);
        assertTrue(people.contains(new Person("Bob", 25)), "got: " + people);
    }

    interface Auditor {
        Result<String> ask(String question);
    }

    @Test
    void resultCarriesToolExecutionsAndAggregatedUsage() {
        Auditor auditor =
                AiServices.builder(Auditor.class)
                        .chatModel(model)
                        .tools(new ServerRoom())
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .build();

        Result<String> result = auditor.ask("What is the server room temperature? One sentence.");
        assertTrue(result.content().contains("41"), result.content());
        assertTrue(
                result.toolExecutions().stream()
                        .anyMatch(e -> e.request().name().equals("temperature")),
                "Result must surface the tool loop's executions");
        // the tool loop aggregates usage via TokenUsage.add - the jinfer subclass (cache
        // accounting, wall times) must survive that aggregation, not decay to the base class
        assertTrue(
                result.tokenUsage() instanceof JinferTokenUsage,
                "aggregated usage lost the jinfer accounting: " + result.tokenUsage());
        assertTrue(result.tokenUsage().totalTokenCount() > 0);
    }
}
