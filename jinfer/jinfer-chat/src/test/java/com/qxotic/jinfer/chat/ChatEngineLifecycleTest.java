package com.qxotic.jinfer.chat;

import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import org.junit.jupiter.api.Test;
import sun.misc.Unsafe;
import java.util.concurrent.locks.ReentrantReadWriteLock;

final class ChatEngineLifecycleTest {

    @Test
    void streamsRunOneAtATimeInSubmissionOrder() throws Exception {
        ChatEngine engine = emptyEngine();
        ThreadPoolExecutor driver = driver(engine);
        List<Integer> order = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch firstStarted = new CountDownLatch(1);
        CountDownLatch releaseFirst = new CountDownLatch(1);
        CountDownLatch secondStarted = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(2);

        try {
            engine.stream(
                    () -> {
                        order.add(1);
                        firstStarted.countDown();
                        await(releaseFirst);
                    });
            assertTrue(firstStarted.await(5, SECONDS));
            engine.stream(
                    () -> {
                        order.add(2);
                        secondStarted.countDown();
                        done.countDown();
                    });
            engine.stream(
                    () -> {
                        order.add(3);
                        done.countDown();
                    });

            assertFalse(secondStarted.await(100, MILLISECONDS));
            releaseFirst.countDown();
            assertTrue(done.await(5, SECONDS));
            assertEquals(List.of(1, 2, 3), order);
        } finally {
            releaseFirst.countDown();
            driver.shutdownNow();
        }
    }

    @Test
    void closeFromStreamDriverThreadFailsInsteadOfWaitingForItself() throws Exception {
        ChatEngine engine = emptyEngine();
        ThreadPoolExecutor driver = driver(engine);
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> thrown = new AtomicReference<>();

        try {
            engine.stream(
                    () -> {
                        try {
                            engine.close();
                        } catch (Throwable t) {
                            thrown.set(t);
                        } finally {
                            done.countDown();
                        }
                    });

            assertTrue(done.await(5, SECONDS), "close() must not wait for its own stream thread");
            assertInstanceOf(IllegalStateException.class, thrown.get());
        } finally {
            driver.shutdownNow();
        }
    }

    @Test
    void interruptedCloseStillWaitsForStreamTermination() throws Exception {
        Arena weights = Arena.ofShared();
        ChatEngine engine = closeReadyEngine(weights);
        ThreadPoolExecutor driver = driver(engine);
        CountDownLatch running = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(1);
        AtomicBoolean interruptRestored = new AtomicBoolean();
        AtomicReference<Throwable> thrown = new AtomicReference<>();

        engine.stream(
                () -> {
                    running.countDown();
                    await(release);
                });
        assertTrue(running.await(5, SECONDS));

        Thread closer =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    try {
                                        engine.close();
                                        interruptRestored.set(
                                                Thread.currentThread().isInterrupted());
                                    } catch (Throwable t) {
                                        thrown.set(t);
                                    } finally {
                                        returned.countDown();
                                    }
                                });
        try {
            awaitShutdown(driver);
            closer.interrupt();

            assertFalse(
                    returned.await(100, MILLISECONDS),
                    "interruption must not weaken close() quiescence");
            assertTrue(weights.scope().isAlive(), "weights freed while the stream was active");

            release.countDown();
            assertTrue(returned.await(5, SECONDS));
            assertNull(thrown.get());
            assertTrue(interruptRestored.get());
            assertFalse(weights.scope().isAlive());
        } finally {
            release.countDown();
            closer.join(SECONDS.toMillis(5));
            driver.shutdownNow();
            Arenas.close(weights);
        }
    }

    @Test
    void concurrentCloseCallsBothWaitForStreamTermination() throws Exception {
        Arena weights = Arena.ofShared();
        ChatEngine engine = closeReadyEngine(weights);
        ThreadPoolExecutor driver = driver(engine);
        CountDownLatch running = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(2);

        engine.stream(
                () -> {
                    running.countDown();
                    await(release);
                });
        assertTrue(running.await(5, SECONDS));

        Thread first = Thread.ofPlatform().start(() -> close(engine, returned));
        Thread second = Thread.ofPlatform().start(() -> close(engine, returned));
        try {
            awaitShutdown(driver);
            assertFalse(returned.await(100, MILLISECONDS));
            assertTrue(weights.scope().isAlive());

            release.countDown();
            assertTrue(returned.await(5, SECONDS));
            assertFalse(weights.scope().isAlive());
        } finally {
            release.countDown();
            first.join(SECONDS.toMillis(5));
            second.join(SECONDS.toMillis(5));
            driver.shutdownNow();
            Arenas.close(weights);
        }
    }

    @Test
    void closeWaitsForQueuedStreamWorkToo() throws Exception {
        Arena weights = Arena.ofShared();
        ChatEngine engine = closeReadyEngine(weights);
        ThreadPoolExecutor driver = driver(engine);
        CountDownLatch firstStarted = new CountDownLatch(1);
        CountDownLatch releaseFirst = new CountDownLatch(1);
        CountDownLatch secondStarted = new CountDownLatch(1);
        CountDownLatch releaseSecond = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(1);

        engine.stream(
                () -> {
                    firstStarted.countDown();
                    await(releaseFirst);
                });
        assertTrue(firstStarted.await(5, SECONDS));
        engine.stream(
                () -> {
                    secondStarted.countDown();
                    await(releaseSecond);
                });
        Thread closer = Thread.ofPlatform().start(() -> close(engine, returned));
        try {
            awaitShutdown(driver);
            releaseFirst.countDown();
            assertTrue(secondStarted.await(5, SECONDS));
            assertFalse(returned.await(100, MILLISECONDS));
            assertTrue(weights.scope().isAlive());

            releaseSecond.countDown();
            assertTrue(returned.await(5, SECONDS));
            assertFalse(weights.scope().isAlive());
        } finally {
            releaseFirst.countDown();
            releaseSecond.countDown();
            closer.join(SECONDS.toMillis(5));
            driver.shutdownNow();
            Arenas.close(weights);
        }
    }

    @Test
    void streamsAreRejectedAsSoonAsCloseStarts() throws Exception {
        Arena weights = Arena.ofShared();
        ChatEngine engine = closeReadyEngine(weights);
        ThreadPoolExecutor driver = driver(engine);
        CountDownLatch running = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch returned = new CountDownLatch(1);

        engine.stream(
                () -> {
                    running.countDown();
                    await(release);
                });
        assertTrue(running.await(5, SECONDS));
        Thread closer = Thread.ofPlatform().start(() -> close(engine, returned));
        try {
            awaitShutdown(driver);
            IllegalStateException rejected =
                    assertThrows(IllegalStateException.class, () -> engine.stream(() -> {}));
            assertEquals("the model is closed", rejected.getMessage());
        } finally {
            release.countDown();
            assertTrue(returned.await(5, SECONDS));
            closer.join(SECONDS.toMillis(5));
            driver.shutdownNow();
            Arenas.close(weights);
        }
    }

    @Test
    void closeFromGenerationCallbackFailsBeforeFreeingResources() throws Exception {
        Arena weights = Arena.ofShared();
        ChatEngine engine = closeReadyEngine(weights);
        ReentrantLock lock = lock(engine);

        lock.lock();
        try {
            assertThrows(IllegalStateException.class, engine::close);
            assertTrue(weights.scope().isAlive());
        } finally {
            lock.unlock();
            engine.close();
            Arenas.close(weights);
        }
    }

    @Test
    void closeIsIdempotentAfterTermination() throws Exception {
        Arena weights = Arena.ofShared();
        ChatEngine engine = closeReadyEngine(weights);

        engine.close();
        engine.close();

        assertFalse(weights.scope().isAlive());
    }

    @Test
    void closeWaitsForPreparationBeforeFreeingWeights() throws Exception {
        Arena weights = Arena.ofShared();
        CountDownLatch preparing = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicBoolean weightsAlive = new AtomicBoolean();
        ChatTemplate template =
                (conversation, batchCapacity, sink) -> {
                    preparing.countDown();
                    await(release);
                    weightsAlive.set(weights.scope().isAlive());
                    throw new IllegalStateException("expected preparation failure");
                };
        ChatEngine engine = preparingEngine(weights, template);
        CountDownLatch closeStarted = new CountDownLatch(1);
        CountDownLatch closeReturned = new CountDownLatch(1);
        AtomicReference<Throwable> preparationFailure = new AtomicReference<>();

        Thread preparation =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    try {
                                        engine.prepare(request());
                                    } catch (Throwable t) {
                                        preparationFailure.set(t);
                                    }
                                });
        assertTrue(preparing.await(5, SECONDS));
        Thread closer =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    closeStarted.countDown();
                                    close(engine, closeReturned);
                                });
        try {
            assertTrue(closeStarted.await(5, SECONDS));
            assertFalse(closeReturned.await(100, MILLISECONDS));
            assertTrue(weights.scope().isAlive());

            release.countDown();
            assertTrue(closeReturned.await(5, SECONDS));
            preparation.join(SECONDS.toMillis(5));
            assertEquals("expected preparation failure", preparationFailure.get().getMessage());
            assertTrue(weightsAlive.get());
            assertFalse(weights.scope().isAlive());
        } finally {
            release.countDown();
            preparation.join(SECONDS.toMillis(5));
            closer.join(SECONDS.toMillis(5));
            driver(engine).shutdownNow();
            Arenas.close(weights);
        }
    }

    private static ChatEngine emptyEngine() throws Exception {
        ChatEngine engine = (ChatEngine) unsafe().allocateInstance(ChatEngine.class);
        set(engine, "streamDriver", newDriver());
        set(engine, "streamThread", new AtomicReference<Thread>());
        set(engine, "lifecycle", new ReentrantReadWriteLock());
        return engine;
    }

    private static ChatEngine closeReadyEngine(Arena weights) throws Exception {
        ChatEngine engine = emptyEngine();
        PromptCache<?> cache = (PromptCache<?>) unsafe().allocateInstance(PromptCache.class);
        Field cacheClosed = PromptCache.class.getDeclaredField("closed");
        cacheClosed.setAccessible(true);
        cacheClosed.setBoolean(cache, true);
        set(engine, "lock", new ReentrantLock(true));
        set(engine, "cache", cache);
        set(engine, "mediaCache", new MediaEncodingCache());
        set(engine, "weights", weights);
        set(engine, "leakWatch", (Runnable) () -> {});
        return engine;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static ChatEngine preparingEngine(Arena weights, ChatTemplate template)
            throws Exception {
        ChatEngine engine = closeReadyEngine(weights);
        LanguageModel model =
                (LanguageModel)
                        Proxy.newProxyInstance(
                                LanguageModel.class.getClassLoader(),
                                new Class<?>[] {LanguageModel.class},
                                (proxy, method, args) -> {
                                    throw new UnsupportedOperationException(method.getName());
                                });
        Tokenizer tokenizer =
                (Tokenizer)
                        Proxy.newProxyInstance(
                                Tokenizer.class.getClassLoader(),
                                new Class<?>[] {Tokenizer.class},
                                (proxy, method, args) -> {
                                    throw new UnsupportedOperationException(method.getName());
                                });
        set(
                engine,
                "loaded",
                new LoadedModel<>(
                        model,
                        tokenizer,
                        "",
                        Set.of(1),
                        ContentKey.sha256(new byte[] {1}),
                        Optional.of(template),
                        LoadedModel.SamplingDefaults.NONE));
        return engine;
    }

    private static ChatEngine.Request request() {
        return ChatEngine.Request.of(List.of(Message.user("hello")), new Sampling(0, 1, 0, 0, 1L));
    }

    private static ThreadPoolExecutor newDriver() {
        return new ThreadPoolExecutor(
                0,
                1,
                60,
                TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(),
                r -> new Thread(r, "jinfer-stream-test"));
    }

    private static ThreadPoolExecutor driver(ChatEngine engine) throws Exception {
        return (ThreadPoolExecutor) field("streamDriver").get(engine);
    }

    private static ReentrantLock lock(ChatEngine engine) throws Exception {
        return (ReentrantLock) field("lock").get(engine);
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
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
    }

    private static void awaitShutdown(ThreadPoolExecutor driver) throws InterruptedException {
        long deadline = System.nanoTime() + SECONDS.toNanos(5);
        while (!driver.isShutdown()) {
            if (System.nanoTime() >= deadline) throw new AssertionError("close() did not start");
            Thread.sleep(1);
        }
    }

    private static Unsafe unsafe() throws Exception {
        Field f = Unsafe.class.getDeclaredField("theUnsafe");
        f.setAccessible(true);
        return (Unsafe) f.get(null);
    }

    private static void set(ChatEngine engine, String name, Object value) throws Exception {
        field(name).set(engine, value);
    }

    private static Field field(String name) throws Exception {
        Field field = ChatEngine.class.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }
}
