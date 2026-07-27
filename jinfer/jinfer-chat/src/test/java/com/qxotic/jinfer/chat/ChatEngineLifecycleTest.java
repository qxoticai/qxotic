package com.qxotic.jinfer.chat;

import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

final class ChatEngineLifecycleTest {

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

    private static ChatEngine emptyEngine() throws Exception {
        Field f = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
        f.setAccessible(true);
        sun.misc.Unsafe unsafe = (sun.misc.Unsafe) f.get(null);
        ChatEngine engine = (ChatEngine) unsafe.allocateInstance(ChatEngine.class);
        set(engine, "streamDriver", newDriver());
        set(engine, "streamThread", new AtomicReference<Thread>());
        return engine;
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

    private static void set(ChatEngine engine, String name, Object value) throws Exception {
        field(name).set(engine, value);
    }

    private static Field field(String name) throws Exception {
        Field field = ChatEngine.class.getDeclaredField(name);
        field.setAccessible(true);
        return field;
    }
}
