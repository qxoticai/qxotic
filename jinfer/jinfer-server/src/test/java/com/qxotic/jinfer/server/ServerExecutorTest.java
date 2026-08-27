package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class ServerExecutorTest {

    @Test
    void saturationAnswers503WithRetryAfterInsteadOfDroppingTheConnection() throws Exception {
        // the bounded executor queue rejected the excess inside the JDK server, which closed
        // the socket with no status; the gate answers like every other overload path
        AtomicInteger served = new AtomicInteger();
        Semaphore admissions = new Semaphore(1);
        var gated = Server.gated(exchange -> served.incrementAndGet(), admissions, 7);

        TestExchange ok = new TestExchange(new byte[0]);
        gated.handle(ok);
        assertEquals(1, served.get());
        assertEquals(1, admissions.availablePermits(), "the permit comes back");

        admissions.acquire();
        TestExchange busy = new TestExchange(new byte[0]);
        gated.handle(busy);
        assertEquals(503, busy.getResponseCode());
        assertEquals("7", busy.getResponseHeaders().getFirst("Retry-After"));
        assertEquals(1, served.get(), "a refused request never reaches the handler");
        assertTrue(Server.requestExecutor(1).getClass().getSimpleName().contains("ThreadPool"));
    }
}
