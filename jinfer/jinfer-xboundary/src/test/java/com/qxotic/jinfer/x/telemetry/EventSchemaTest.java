package com.qxotic.jinfer.x.telemetry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import jdk.jfr.Event;
import jdk.jfr.EventType;
import jdk.jfr.ValueDescriptor;
import org.junit.jupiter.api.Test;

/** Pins the JFR names and fields consumed by recordings, settings and exporters. */
class EventSchemaTest {

    @Test
    void shippedJfrProfileCoversEveryPublicEvent() throws Exception {
        String settings;
        try (var in = EventSchemaTest.class.getResourceAsStream("/jinfer.jfc")) {
            assertNotNull(in, "missing /jinfer.jfc");
            settings = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        for (String name :
                List.of(
                        "jinfer.Inference",
                        "jinfer.ModelLoad",
                        "jinfer.Runtime",
                        "jinfer.PromptCache",
                        "jinfer.MediaCache",
                        "jinfer.MediaProjection",
                        "jinfer.Speculation",
                        "jinfer.Decode")) {
            assertTrue(settings.contains("event name=\"" + name + "\""), name);
        }
    }

    @Test
    void inferenceEventIsStable() {
        assertEvent(
                InferenceEvent.class,
                "jinfer.Inference",
                List.of("jinfer", "Inference"),
                Set.of(
                        "model",
                        "operation",
                        "outputType",
                        "inputTokens",
                        "outputTokens",
                        "reasoningTokens",
                        "cachedTokens",
                        "cacheTier",
                        "queueTime",
                        "prefillTime",
                        "decodeTime",
                        "timeToFirstToken",
                        "finishReason",
                        "errorType"));
    }

    @Test
    void lifecycleEventsAreStable() {
        assertEvent(
                ModelLoadEvent.class,
                "jinfer.ModelLoad",
                List.of("jinfer", "Lifecycle"),
                Set.of(
                        "model",
                        "architecture",
                        "contextLength",
                        "dimensions",
                        "weightsBytes",
                        "mapped"));
        assertEvent(
                RuntimeEvent.class,
                "jinfer.Runtime",
                List.of("jinfer", "Lifecycle"),
                Set.of("vectorBits", "decodeThreads"));
    }

    @Test
    void generationEventsAreStable() {
        assertEvent(
                SpeculationEvent.class,
                "jinfer.Speculation",
                List.of("jinfer", "Inference"),
                Set.of("draftedTokens", "acceptedTokens", "forwards"));
        assertEvent(DecodeEvent.class, "jinfer.Decode", List.of("jinfer", "Inference"), Set.of());
        assertTrue(!EventType.getEventType(DecodeEvent.class).isEnabled());
    }

    @Test
    void cacheEventsAreStable() {
        assertEvent(
                PromptCacheEvent.class,
                "jinfer.PromptCache",
                List.of("jinfer", "Memory"),
                Set.of(
                        "model",
                        "retainedSessions",
                        "retainedSessionLimit",
                        "sessionHits",
                        "stateAllocations",
                        "sessionSnapshotBytes",
                        "blocks",
                        "bytes",
                        "budgetBytes",
                        "blockHits",
                        "blockMisses",
                        "blockEvictions",
                        "blockDiscards",
                        "blockRefusals"));
        assertEvent(
                MediaCacheEvent.class,
                "jinfer.MediaCache",
                List.of("jinfer", "Memory"),
                Set.of("model", "entries", "bytes", "budgetBytes", "hits", "misses", "refusals"));
    }

    @Test
    void mediaProjectionEventIsStable() {
        assertEvent(
                MediaProjectionEvent.class,
                "jinfer.MediaProjection",
                List.of("jinfer", "Inference"),
                Set.of(
                        "modality",
                        "sourceWidth",
                        "sourceHeight",
                        "sourceChannels",
                        "sampledFrames",
                        "sourceSampleRate",
                        "sourceDuration",
                        "errorType"));
    }

    @Test
    void labelsFollowJfrConventions() {
        for (Class<? extends Event> event :
                List.of(
                        InferenceEvent.class,
                        ModelLoadEvent.class,
                        RuntimeEvent.class,
                        SpeculationEvent.class,
                        DecodeEvent.class,
                        PromptCacheEvent.class,
                        MediaCacheEvent.class,
                        MediaProjectionEvent.class)) {
            EventType type = EventType.getEventType(event);
            checkLabel(type.getLabel(), type.getName());
            for (ValueDescriptor field : type.getFields()) {
                if (declared(type).contains(field.getName())) {
                    checkLabel(field.getLabel(), type.getName() + "#" + field.getName());
                }
            }
        }
    }

    private static void assertEvent(
            Class<? extends Event> event,
            String name,
            List<String> categories,
            Set<String> fields) {
        EventType type = EventType.getEventType(event);
        assertEquals(name, type.getName());
        assertEquals(categories, type.getCategoryNames());
        assertEquals(fields, declared(type));
    }

    private static void checkLabel(String label, String what) {
        assertNotNull(label, what + " has no @Label");
        assertTrue(Character.isUpperCase(label.charAt(0)), what + " label must start uppercase");
        assertTrue(!label.endsWith("."), what + " label must not end with punctuation");
        assertTrue(!label.contains("Event"), what + " label must omit Event");
    }

    private static Set<String> declared(EventType type) {
        Set<String> builtIn = Set.of("startTime", "duration", "eventThread", "stackTrace");
        return type.getFields().stream()
                .map(ValueDescriptor::getName)
                .filter(name -> !builtIn.contains(name))
                .collect(Collectors.toSet());
    }
}
