package com.qxotic.jinfer.telemetry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import java.time.Duration;
import org.junit.jupiter.api.Test;

/**
 * The registry samples live caches forever, so holding them strongly would pin every engine a
 * process ever built - the exact shape of a slow leak in a long-running server. It holds weak
 * references instead, which only works if the registrant keeps the gauge in a field. Both halves of
 * that contract are pinned here.
 */
class CacheGaugeRegistryTest {

    @Test
    void aGaugeIsSampledWhileItsOwnerLivesAndDroppedAfterwards() {
        int before = Telemetry.liveGauges();

        Telemetry.CacheGauge held =
                new Telemetry.CacheGauge("held", () -> new PromptCache.Sample(1, 2, 3, 4, 5, 6));
        Telemetry.register(held);
        assertEquals(before + 1, Telemetry.liveGauges(), "a held gauge must stay registered");

        // exactly what a caller must NOT do - registered and immediately unreachable
        Telemetry.register(
                new Telemetry.CacheGauge(
                        "dropped", () -> new PromptCache.Sample(0, 0, 0, 0, 0, 0)));

        assertTrue(
                await(() -> Telemetry.liveGauges() == before + 1),
                "an unreferenced gauge must be collected, not accumulate in the registry");

        // the held one is still there only because this frame still references it
        assertEquals("held", held.model());
    }

    /** GC is not synchronous; poll rather than assert once. */
    private static boolean await(java.util.function.BooleanSupplier condition) {
        long deadline = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (System.nanoTime() < deadline) {
            System.gc();
            if (condition.getAsBoolean()) return true;
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return false;
    }
}
