package com.qxotic.jinfer.telemetry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import java.time.Duration;
import org.junit.jupiter.api.Test;

/**
 * Registration is deterministic - register on construction, unregister on close - and the weak
 * reference behind it is only a backstop for an owner that is never closed.
 *
 * <p>That backstop is not optional politeness. {@code LeakWatch} reports unclosed engines from a
 * {@link java.lang.ref.Cleaner}, so a registry holding owners strongly would pin them forever, the
 * Cleaner would never fire, and adding telemetry would have silently switched off jinfer's own leak
 * detection.
 */
class CacheGaugeRegistryTest {

    private static Telemetry.CacheGauge gauge(String model) {
        return new Telemetry.CacheGauge(model, () -> new PromptCache.Sample(1, 2, 3, 4, 5, 6, 0));
    }

    @Test
    void unregisterStopsSamplingImmediately() {
        int before = Telemetry.liveGauges();
        Telemetry.CacheGauge gauge = gauge("closed-properly");

        Telemetry.register(gauge);
        assertEquals(before + 1, Telemetry.liveGauges());

        Telemetry.unregister(gauge);
        assertEquals(
                before, Telemetry.liveGauges(), "close must deregister without waiting for GC");

        Telemetry.unregister(gauge); // idempotent, like every close here
        assertEquals(before, Telemetry.liveGauges());
    }

    @Test
    void anOwnerThatIsNeverClosedStaysCollectable() {
        int before = Telemetry.liveGauges();
        Telemetry.register(gauge("never-closed"));

        assertTrue(
                await(() -> Telemetry.liveGauges() == before),
                "an unreferenced gauge must not pin its owner - that would stop LeakWatch firing");
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
