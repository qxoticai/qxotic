package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@code AutoCloseable} semantics against the small LFM2 (cheap to load per test) - the mirror of
 * the spring twin's lifecycle contract. Model-gated: assume-skips when the file is absent.
 */
@Tag("integration")
class JinferLifecycleIT {

    static final Path SMALL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModelSmall", ModelFixture.LFM25_350M_Q8.path().toString()));

    private static JinferChatModel load() {
        return JinferChatModel.builder()
                .modelPath(SMALL)
                .contextLength(2048)
                .maxOutputTokens(8)
                .build();
    }

    @Test
    void closeGuardsEveryEntryPointAndIsIdempotent() throws Exception {
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        JinferChatModel m = load();
        m.chat(UserMessage.from("hi")); // proves the model worked before close
        m.close();
        m.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> m.chat(UserMessage.from("hi")));
        assertThrows(
                IllegalStateException.class,
                () ->
                        m.streaming()
                                .chat(
                                        "hi",
                                        new dev.langchain4j.model.chat.response
                                                .StreamingChatResponseHandler() {
                                            @Override
                                            public void onPartialResponse(String partial) {}

                                            @Override
                                            public void onCompleteResponse(
                                                    dev.langchain4j.model.chat.response.ChatResponse
                                                            response) {}

                                            @Override
                                            public void onError(Throwable error) {}
                                        }));
        assertThrows(
                IllegalStateException.class,
                () -> m.withCachedPrompt(List.of(SystemMessage.from("x")), List.of()));
        assertThrows(
                IllegalStateException.class,
                () -> m.saveCachedPrompts(Path.of("/tmp/jinfer-closed.jkv")));
    }

    @Test
    void closeDuringALiveStreamWaitsForItInsteadOfCrashing() throws Exception {
        // the arena law made executable: close() must not free state memory while the stream
        // driver is mid-generation - it blocks (engine lock + driver await), the stream finishes
        // or fails cleanly, and only then the pooled states' arenas die
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        JinferChatModel m = load();
        var done = new java.util.concurrent.CountDownLatch(1);
        var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
        var sawToken = new java.util.concurrent.CountDownLatch(1);
        m.streaming()
                .chat(
                        "Count from 1 to 30.",
                        new dev.langchain4j.model.chat.response.StreamingChatResponseHandler() {
                            @Override
                            public void onPartialResponse(String partial) {
                                sawToken.countDown();
                            }

                            @Override
                            public void onPartialThinking(
                                    dev.langchain4j.model.chat.response.PartialThinking partial) {
                                sawToken.countDown(); // thinking families stream these first
                            }

                            @Override
                            public void onCompleteResponse(
                                    dev.langchain4j.model.chat.response.ChatResponse response) {
                                done.countDown();
                            }

                            @Override
                            public void onError(Throwable error) {
                                failure.set(error);
                                done.countDown();
                            }
                        });
        // provably mid-generation (any delta); a capped wait so a silent family can't hang us
        sawToken.await(60, java.util.concurrent.TimeUnit.SECONDS);
        m.close(); // must BLOCK until the generation completed - never a crash
        org.junit.jupiter.api.Assertions.assertTrue(
                done.await(30, java.util.concurrent.TimeUnit.SECONDS),
                "stream neither completed nor failed after close returned");
        if (failure.get() != null) {
            org.junit.jupiter.api.Assertions.assertInstanceOf(
                    IllegalStateException.class, failure.get(), String.valueOf(failure.get()));
        }
    }

    @Test
    void requestsReuseOneContextInsteadOfAllocatingPerGeneration() {
        // the pool IS the allocator: the first request allocates this pipeline's context, every
        // later one reuses it (reset between requests). N requests cost ONE context, and the
        // stateless default never keeps a conversation across them.
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        try (JinferChatModel model = load()) {
            for (int i = 0; i < 5; i++) {
                model.chat(UserMessage.from("say hi to guest " + i));
            }
            String stats = model.engine.sessionStats();
            assertTrue(stats.contains("allocations=1"), "one context for five requests: " + stats);
            assertTrue(stats.contains("hits=0"), "the default matches nothing: " + stats);
            // the allocation survives as the pool's wiped spare, not as a live session: the
            // stateless default keeps the memory and none of the conversation
            assertTrue(stats.contains("sessions=0"), "the default pools no conversation: " + stats);
        }
    }

    @Test
    void theStatelessDefaultKeepsNoConversationBetweenRequests() {
        // "nothing persists" has to mean the KV too: the pooled allocation is wiped when its
        // reply ends, so an unrelated request cannot see the previous conversation - and an
        // identical request must re-prefill rather than extend anything
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        try (JinferChatModel model = load()) {
            String first = model.chat(UserMessage.from("name a colour")).aiMessage().text();
            model.chat(UserMessage.from("name a country"));
            String again = model.chat(UserMessage.from("name a colour")).aiMessage().text();
            assertEquals(first, again, "a greedy request must not depend on what ran before it");
            assertTrue(
                    model.engine.sessionStats().contains("hits=0"),
                    "the stateless default must never serve from a kept conversation");
        }
    }

    @Test
    void repeatedLoadChatCloseIsFootprintBounded() throws Exception {
        // the leak gate that would have caught the 51GB battery OOM: every cycle frees its
        // states deterministically at close, so N cycles cost ~one model, not N
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        long before = rssKb();
        for (int i = 0; i < 6; i++) {
            JinferChatModel m = load();
            m.chat(UserMessage.from("hi"));
            m.close();
        }
        long grownMb = (rssKb() - before) / 1024;
        // one 350M state is ~hundreds of MB; 6 leaked cycles would blow far past this bound
        org.junit.jupiter.api.Assertions.assertTrue(
                grownMb < 1500, "RSS grew " + grownMb + " MB over 6 load/chat/close cycles");
    }

    private static long rssKb() throws Exception {
        for (String line : Files.readAllLines(Path.of("/proc/self/status"))) {
            if (line.startsWith("VmRSS:")) {
                return Long.parseLong(line.replaceAll("[^0-9]", ""));
            }
        }
        throw new IllegalStateException("no VmRSS");
    }

    @Test
    void closingTheBaseClosesViews() {
        Assumptions.assumeTrue(Files.exists(SMALL), "model not found: " + SMALL);
        JinferChatModel m = load();
        JinferChatModel view =
                m.withCachedPrompt(List.of(SystemMessage.from("You are terse.")), List.of());
        m.close();
        assertThrows(IllegalStateException.class, () -> view.chat(UserMessage.from("hi")));
    }
}
