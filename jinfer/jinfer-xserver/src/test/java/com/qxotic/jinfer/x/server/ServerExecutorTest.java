package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class ServerExecutorTest {

    @Test
    void boundsActiveAndWaitingHttpRequests() throws Exception {
        ExecutorService executor = Server.requestExecutor(1);
        CountDownLatch running = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch queuedRan = new CountDownLatch(1);
        try {
            executor.execute(
                    () -> {
                        running.countDown();
                        await(release);
                    });
            assertTrue(running.await(2, TimeUnit.SECONDS));
            executor.execute(queuedRan::countDown);

            assertThrows(RejectedExecutionException.class, () -> executor.execute(() -> {}));
            release.countDown();
            assertTrue(queuedRan.await(2, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            executor.shutdownNow();
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
