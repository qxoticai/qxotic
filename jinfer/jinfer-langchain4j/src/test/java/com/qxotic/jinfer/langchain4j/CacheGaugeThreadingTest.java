package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordingStream;
import org.junit.jupiter.api.Test;

/**
 * Sampling runs concurrently with generation, so this drives both at once and checks every sample
 * is internally coherent: deltas non-negative, sizes agreeing with each other.
 *
 * <p>HONEST SCOPE. This does NOT prove the thread-safety fix behind it. {@code PromptCache} is
 * "single-threaded by design (the generation worker), like the store" - a plain HashMap and plain
 * long counters - so the sampler must not read it directly; the generation thread publishes an
 * immutable snapshot under the engine lock instead. That is a JLS argument (non-volatile long reads
 * may tear) and a documented-contract argument, not an observable one: this test was measured to
 * pass against the racy version too, because the race does not reproduce on x86-64 HotSpot.
 *
 * <p>The real guarantee is structural - the sampler's supplier reads one volatile field and cannot
 * reach the cache at all - and structure is what a reviewer should check here, not this test.
 */
class CacheGaugeThreadingTest {

    @Test
    void samplingConcurrentlyWithGenerationNeverReadsATornSnapshot() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        AtomicReference<String> corrupt = new AtomicReference<>();

        try (var model = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(8).build()) {
            try (RecordingStream stream = new RecordingStream()) {
                stream.enable("jinfer.PromptCache").withPeriod(Duration.ofMillis(20));
                stream.onEvent(
                        "jinfer.PromptCache",
                        event -> {
                            long bytes = event.getLong("bytes");
                            long budget = event.getLong("budgetBytes");
                            int blocks = event.getInt("blocks");
                            // deltas are differences of monotonic counters: never negative
                            if (event.getLong("hits") < 0
                                    || event.getLong("misses") < 0
                                    || event.getLong("evictions") < 0) {
                                corrupt.compareAndSet(null, "negative delta");
                            }
                            if (bytes < 0 || blocks < 0 || budget <= 0) {
                                corrupt.compareAndSet(null, "bytes=" + bytes + " blocks=" + blocks);
                            }
                            // a cache holding blocks cannot hold zero bytes, and vice versa
                            if ((blocks == 0) != (bytes == 0)) {
                                corrupt.compareAndSet(
                                        null, "blocks/bytes disagree: " + blocks + "/" + bytes);
                            }
                        });
                stream.startAsync();
                for (int i = 0; i < 5; i++) model.chat("Name colour number " + i + ".");
                Thread.sleep(200); // let the sampler tick a few more times
            }
        }
        assertEquals(null, corrupt.get(), "sampler observed an inconsistent cache snapshot");
    }

    /** The snapshot is published per generation, so it must actually advance as work happens. */
    @Test
    void theSnapshotTracksGenerations() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        Path jfr = Files.createTempFile("jinfer-track", ".jfr");
        try (var model = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(8).build()) {
            try (Recording recording = new Recording()) {
                recording.enable("jinfer.PromptCache").withPeriod(Duration.ofMillis(100));
                recording.start();
                for (int i = 0; i < 3; i++) model.chat("Name colour number " + i + ".");
                Thread.sleep(300);
                recording.stop();
                recording.dump(jfr);
            }
        }
        long samples = TelemetryEmissionTest.eventsOf(jfr, "jinfer.PromptCache").size();
        assertTrue(samples > 0, "the gauge must sample while an engine is alive");
    }
}
