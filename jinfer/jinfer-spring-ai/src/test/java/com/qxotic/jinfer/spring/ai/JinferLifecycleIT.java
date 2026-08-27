package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * {@code AutoCloseable} semantics against the small LFM2.5 (cheap to load per test). Model-gated
 * via {@link TestModels}. Run: {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl
 * jinfer-spring-ai}
 */
@Tag("integration")
class JinferLifecycleIT {

    static final Path SMALL =
            TestModels.require("hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf");

    private static JinferChatModel load() {
        return JinferChatModel.builder()
                .modelPath(SMALL)
                .contextLength(2048)
                .options(JinferChatOptions.builder().maxTokens(8).build())
                .build();
    }

    @Test
    void aStreamQueuedBehindAnotherFailsLoudlyWhenTheModelCloses() throws Exception {
        // the queued stream's prepare runs on the driver thread after close(): it must reach the
        // sink as an error, not die uncaught and leave the subscriber waiting forever
        JinferChatModel m = load();
        Prompt slow =
                new Prompt(
                        new UserMessage("Write a long story about a lighthouse keeper."),
                        JinferChatOptions.builder().maxTokens(300).build());
        Prompt quick = new Prompt(new UserMessage("hi"));
        java.util.concurrent.CountDownLatch firstChunk = new java.util.concurrent.CountDownLatch(1);
        m.stream(slow).subscribe(chunk -> firstChunk.countDown(), error -> {}, () -> {});
        assertTrue(firstChunk.await(30, java.util.concurrent.TimeUnit.SECONDS));
        java.util.concurrent.CompletableFuture<Throwable> queued =
                new java.util.concurrent.CompletableFuture<>();
        m.stream(quick)
                .subscribe(
                        chunk -> {},
                        queued::complete,
                        () -> queued.complete(new AssertionError("completed after close")));
        Thread closer = new Thread(m::close, "closer");
        closer.start();
        Throwable outcome = queued.get(60, java.util.concurrent.TimeUnit.SECONDS);
        closer.join();
        assertTrue(outcome instanceof IllegalStateException, String.valueOf(outcome));
    }

    @Test
    void closeGuardsEveryEntryPointAndIsIdempotent() {
        JinferChatModel m = load();
        m.call(new Prompt(new UserMessage("hi"))); // proves the model worked before close
        m.close();
        m.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> m.call(new Prompt(new UserMessage("hi"))));
        assertThrows(
                IllegalStateException.class,
                () -> m.stream(new Prompt(new UserMessage("hi"))).blockLast(Duration.ofMinutes(1)));
        assertThrows(
                IllegalStateException.class,
                () -> m.withCachedPrompt(List.of(new SystemMessage("x")), List.of()));
        assertThrows(
                IllegalStateException.class,
                () -> m.saveCachedPrompts(Path.of("/tmp/opencode/x.jkv")));
    }

    @Test
    void closingTheBaseClosesViews() {
        JinferChatModel m = load();
        JinferChatModel view =
                m.withCachedPrompt(List.of(new SystemMessage("You are terse.")), List.of());
        m.close();
        assertThrows(
                IllegalStateException.class, () -> view.call(new Prompt(new UserMessage("hi"))));
    }
}
