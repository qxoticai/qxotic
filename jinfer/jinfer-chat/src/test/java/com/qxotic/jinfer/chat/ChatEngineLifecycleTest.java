package com.qxotic.jinfer.chat;

import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Isolated;

/** Real engines and owned arenas; latches control the work rather than private-field injection. */
@Isolated("Observes the test provider's last owned arena")
final class ChatEngineLifecycleTest {
    @TempDir Path directory;

    private ChatEngine ownedEngine() throws Exception {
        Path file = directory.resolve("fake.gguf");
        GGUF.write(Builder.newBuilder().putString("general.architecture", "fake").build(), file);
        return new ChatEngine(file, Map.of(), PromptCache.Options.DEFAULTS.withBlockBudget(0));
    }

    @Test
    void streamsRunOneAtATimeInSubmissionOrder() throws Exception {
        CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        CountDownLatch second = new CountDownLatch(1), done = new CountDownLatch(2);
        List<Integer> order = Collections.synchronizedList(new ArrayList<>());
        try (ChatEngine engine = ownedEngine()) {
            try {
                engine.stream(
                        () -> {
                            order.add(1);
                            entered.countDown();
                            await(release);
                        });
                assertTrue(entered.await(5, SECONDS));
                engine.stream(
                        () -> {
                            order.add(2);
                            second.countDown();
                            done.countDown();
                        });
                engine.stream(
                        () -> {
                            order.add(3);
                            done.countDown();
                        });
                assertFalse(second.await(100, MILLISECONDS));
            } finally {
                release.countDown();
            }
            assertTrue(done.await(5, SECONDS));
            assertEquals(List.of(1, 2, 3), order);
        }
    }

    @Test
    void closeFromStreamDriverThreadFailsInsteadOfWaitingForItself() throws Exception {
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        try (ChatEngine engine = ownedEngine()) {
            engine.stream(
                    () -> {
                        try {
                            engine.close();
                        } catch (Throwable t) {
                            failure.set(t);
                        } finally {
                            done.countDown();
                        }
                    });
            assertTrue(done.await(5, SECONDS));
            assertInstanceOf(IllegalStateException.class, failure.get());
        }
    }

    @Test
    void interruptedCloseStillWaitsForStreamTermination() throws Exception {
        CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(1);
        AtomicBoolean restored = new AtomicBoolean();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        try (ChatEngine engine = ownedEngine()) {
            Arena weights = RecordingProvider.last().arena();
            engine.stream(
                    () -> {
                        entered.countDown();
                        await(release);
                    });
            assertTrue(entered.await(5, SECONDS));
            Thread closer =
                    Thread.ofPlatform()
                            .start(
                                    () -> {
                                        try {
                                            engine.close();
                                            restored.set(Thread.currentThread().isInterrupted());
                                        } catch (Throwable t) {
                                            failure.set(t);
                                        } finally {
                                            returned.countDown();
                                        }
                                    });
            try {
                awaitClosing(engine);
                closer.interrupt();
                assertFalse(returned.await(100, MILLISECONDS));
                assertTrue(weights.scope().isAlive());
            } finally {
                release.countDown();
                join(closer);
            }
            assertNull(failure.get());
            assertTrue(restored.get());
            assertFalse(weights.scope().isAlive());
        }
    }

    @Test
    void concurrentCloseCallsBothWaitForStreamTermination() throws Exception {
        CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(2);
        try (ChatEngine engine = ownedEngine()) {
            Arena weights = RecordingProvider.last().arena();
            engine.stream(
                    () -> {
                        entered.countDown();
                        await(release);
                    });
            assertTrue(entered.await(5, SECONDS));
            Thread first = Thread.ofPlatform().start(() -> close(engine, returned));
            Thread second = Thread.ofPlatform().start(() -> close(engine, returned));
            try {
                awaitClosing(engine);
                assertFalse(returned.await(100, MILLISECONDS));
                assertTrue(weights.scope().isAlive());
            } finally {
                release.countDown();
                join(first);
                join(second);
            }
            assertEquals(0, returned.getCount());
            assertFalse(weights.scope().isAlive());
        }
    }

    @Test
    void closeWaitsForQueuedStreamWorkToo() throws Exception {
        CountDownLatch first = new CountDownLatch(1), releaseFirst = new CountDownLatch(1);
        CountDownLatch second = new CountDownLatch(1), releaseSecond = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(1);
        try (ChatEngine engine = ownedEngine()) {
            Arena weights = RecordingProvider.last().arena();
            engine.stream(
                    () -> {
                        first.countDown();
                        await(releaseFirst);
                    });
            assertTrue(first.await(5, SECONDS));
            engine.stream(
                    () -> {
                        second.countDown();
                        await(releaseSecond);
                    });
            Thread closer = Thread.ofPlatform().start(() -> close(engine, returned));
            try {
                awaitClosing(engine);
                releaseFirst.countDown();
                assertTrue(second.await(5, SECONDS));
                assertFalse(returned.await(100, MILLISECONDS));
                assertTrue(weights.scope().isAlive());
            } finally {
                releaseFirst.countDown();
                releaseSecond.countDown();
                join(closer);
            }
            assertFalse(weights.scope().isAlive());
        }
    }

    @Test
    void streamsAreRejectedAsSoonAsCloseStarts() throws Exception {
        CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        try (ChatEngine engine = ownedEngine()) {
            engine.stream(
                    () -> {
                        entered.countDown();
                        await(release);
                    });
            assertTrue(entered.await(5, SECONDS));
            Thread closer = Thread.ofPlatform().start(engine::close);
            try {
                awaitClosing(engine);
                var rejected =
                        assertThrows(IllegalStateException.class, () -> engine.stream(() -> {}));
                assertEquals("the model is closed", rejected.getMessage());
            } finally {
                release.countDown();
                join(closer);
            }
        }
    }

    @Test
    void closeFromGenerationCallbackFailsBeforeFreeingResources() throws Exception {
        try (ChatEngine engine = ownedEngine()) {
            Arena weights = RecordingProvider.last().arena();
            AtomicBoolean called = new AtomicBoolean();
            engine.generate(
                    List.of(Batch.step('a')),
                    Sampler.ARGMAX,
                    1,
                    Duration.ZERO,
                    token -> {
                        called.set(true);
                        assertThrows(IllegalStateException.class, engine::close);
                        assertTrue(weights.scope().isAlive());
                        return true;
                    });
            assertTrue(called.get());
        }
    }

    @Test
    void closeIsIdempotentAfterTermination() throws Exception {
        ChatEngine engine = ownedEngine();
        Arena weights = RecordingProvider.last().arena();
        engine.close();
        engine.close();
        assertFalse(weights.scope().isAlive());
    }

    @Test
    void closeWaitsForPreparationBeforeTheCallerCanFreeWeights() throws Exception {
        CountDownLatch preparing = new CountDownLatch(1), release = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        try (Arena weights = Arena.ofShared()) {
            ChatTemplate template =
                    (conversation, capacity, sink) -> {
                        preparing.countDown();
                        await(release);
                        assertTrue(weights.scope().isAlive());
                        throw new IllegalStateException("expected preparation failure");
                    };
            try (ChatEngine engine = engine(template)) {
                Thread prepare =
                        Thread.ofPlatform()
                                .start(
                                        () -> {
                                            try {
                                                engine.prepare(request(false, -1, null));
                                            } catch (Throwable t) {
                                                failure.set(t);
                                            }
                                        });
                assertTrue(preparing.await(5, SECONDS));
                Thread closer = Thread.ofPlatform().start(() -> close(engine, returned));
                try {
                    assertFalse(returned.await(100, MILLISECONDS));
                } finally {
                    release.countDown();
                    join(prepare);
                    join(closer);
                }
                assertEquals("expected preparation failure", failure.get().getMessage());
                assertTrue(weights.scope().isAlive(), "borrowed weights stay caller-owned");
            }
        }
    }

    @Test
    void anAlwaysReasoningCheckpointRefusesThinkingOffBeforeEncoding() {
        AtomicBoolean encoded = new AtomicBoolean();
        AtomicReference<Boolean> thinking = new AtomicReference<>();
        ChatTemplate template =
                new ChatTemplate() {
                    public ReplyState encode(
                            Conversation conversation, int capacity, Consumer<Batch> sink) {
                        encoded.set(true);
                        thinking.set(conversation.thinking());
                        throw new IllegalStateException("expected: encode reached");
                    }

                    public ThinkingPolicy thinkingPolicy() {
                        return ThinkingPolicy.ALWAYS;
                    }
                };
        try (ChatEngine engine = engine(template)) {
            assertEquals(ChatTemplate.ThinkingPolicy.ALWAYS, engine.thinkingPolicy());
            var off =
                    assertThrows(
                            UnsupportedOperationException.class,
                            () -> engine.prepare(request(false, -1, null)));
            assertTrue(off.getMessage().contains("always reasons"));
            assertTrue(off.getMessage().contains("thinking off"));
            assertFalse(encoded.get());
            assertThrows(IllegalStateException.class, () -> engine.prepare(request(true, -1, 48)));
            assertTrue(encoded.get());
            thinking.set(null);
            assertThrows(
                    IllegalStateException.class,
                    () -> engine.prepare(request(true, ChatEngine.THINK_FLOOR - 1, null)));
            assertEquals(Boolean.TRUE, thinking.get());
        }
    }

    private static ChatEngine engine(ChatTemplate template) {
        var loaded =
                new LoadedModel<>(
                        new TestLanguageModel(),
                        TestLanguageModel.TOKENIZER,
                        "",
                        Set.of(),
                        new ContentKey("lifecycle-test"),
                        Optional.of(template),
                        LoadedModel.SamplingDefaults.NONE);
        return new ChatEngine(loaded, "test", PromptCache.Options.DEFAULTS.withBlockBudget(0));
    }

    private static ChatEngine.Request request(
            boolean thinking, int maxOutputTokens, Integer budget) {
        return new ChatEngine.Request(
                List.of(Message.user("hello")),
                List.of(),
                thinking,
                maxOutputTokens,
                budget,
                null,
                Duration.ZERO,
                new Sampling(0, 1, 0, 0, 1L),
                null,
                null,
                List.of(),
                null);
    }

    private static void close(ChatEngine engine, CountDownLatch returned) {
        try {
            engine.close();
        } finally {
            returned.countDown();
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(10, SECONDS), "work was not released");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
    }

    private static void join(Thread thread) throws InterruptedException {
        thread.join(SECONDS.toMillis(5));
        assertFalse(thread.isAlive(), "thread did not finish: " + thread.getName());
    }

    private static void awaitClosing(ChatEngine engine) throws InterruptedException {
        long deadline = System.nanoTime() + SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            try {
                engine.stream(() -> {});
            } catch (IllegalStateException closed) {
                return;
            }
            Thread.sleep(1);
        }
        fail("close did not stop admission");
    }
}
