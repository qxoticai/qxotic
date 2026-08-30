package com.qxotic.jinfer.telemetry;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;

/** Pins both cache events' complete value mapping and delta accounting. */
class CacheGaugeEmissionTest {

    @Test
    void emitsPromptGaugesAndCounterDeltas() throws Exception {
        CacheSample first = new CacheSample(1, 2, 3, 4, 500, 6, 700, 800, 9, 10, 11, 12, 13);
        CacheSample second = new CacheSample(2, 2, 5, 7, 600, 7, 750, 800, 13, 15, 17, 19, 21);
        AtomicReference<CacheSample> source = new AtomicReference<>(first);
        Telemetry.CacheGauge gauge =
                new Telemetry.CacheGauge("test-model", source::get, () -> null);
        List<RecordedEvent> events =
                record(
                        "jinfer.PromptCache",
                        () -> {
                            gauge.emitPrompt();
                            source.set(second);
                            gauge.emitPrompt();
                        });

        assertEquals(2, events.size());
        assertModel(events);
        assertEquals(first, promptSample(events.get(0)), "first emission is relative to zero");
        assertEquals(
                new CacheSample(2, 2, 2, 3, 600, 7, 750, 800, 4, 5, 6, 7, 8),
                promptSample(events.get(1)),
                "gauges stay absolute; counters become deltas");
    }

    @Test
    void emitsMediaGaugesAndCounterDeltas() throws Exception {
        MediaCacheSample first = new MediaCacheSample(1, 200, 300, 4, 5, 6);
        MediaCacheSample second = new MediaCacheSample(2, 250, 300, 7, 9, 11);
        AtomicReference<MediaCacheSample> source = new AtomicReference<>(first);
        Telemetry.CacheGauge gauge =
                new Telemetry.CacheGauge("test-model", () -> null, source::get);
        List<RecordedEvent> events =
                record(
                        "jinfer.MediaCache",
                        () -> {
                            gauge.emitMedia();
                            source.set(second);
                            gauge.emitMedia();
                        });

        assertEquals(2, events.size());
        assertModel(events);
        assertEquals(first, mediaSample(events.get(0)), "first emission is relative to zero");
        assertEquals(
                new MediaCacheSample(2, 250, 300, 3, 4, 5),
                mediaSample(events.get(1)),
                "gauges stay absolute; counters become deltas");
    }

    private static List<RecordedEvent> record(String name, Runnable emit) throws Exception {
        Path jfr = Files.createTempFile("jinfer-cache-gauge", ".jfr");
        try {
            try (Recording recording = new Recording()) {
                recording.enable(name);
                recording.start();
                emit.run();
                recording.stop();
                recording.dump(jfr);
            }
            return RecordingFile.readAllEvents(jfr).stream()
                    .filter(event -> event.getEventType().getName().equals(name))
                    .toList();
        } finally {
            Files.deleteIfExists(jfr);
        }
    }

    private static void assertModel(List<RecordedEvent> events) {
        for (RecordedEvent event : events) assertEquals("test-model", event.getString("model"));
    }

    private static CacheSample promptSample(RecordedEvent event) {
        return new CacheSample(
                event.getInt("retainedSessions"),
                event.getInt("retainedSessionLimit"),
                event.getLong("sessionHits"),
                event.getLong("stateAllocations"),
                event.getLong("sessionSnapshotBytes"),
                event.getInt("blocks"),
                event.getLong("bytes"),
                event.getLong("budgetBytes"),
                event.getLong("blockHits"),
                event.getLong("blockMisses"),
                event.getLong("blockEvictions"),
                event.getLong("blockDiscards"),
                event.getLong("blockRefusals"));
    }

    private static MediaCacheSample mediaSample(RecordedEvent event) {
        return new MediaCacheSample(
                event.getInt("entries"),
                event.getLong("bytes"),
                event.getLong("budgetBytes"),
                event.getLong("hits"),
                event.getLong("misses"),
                event.getLong("refusals"));
    }
}
