package com.qxotic.jinfer.telemetry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import jdk.jfr.Event;
import jdk.jfr.EventType;
import jdk.jfr.ValueDescriptor;
import org.junit.jupiter.api.Test;

/**
 * The event schema IS public API: names go into users' {@code .jfc} settings files and into every
 * query they write, so renaming one breaks their tooling as surely as renaming a public method -
 * silently, and only at their next recording. This pins it.
 *
 * <p>Cheap by design (no model, no recording), because its job is to fail the build the moment a
 * refactor drifts, which is the usual way telemetry rots.
 */
class EventSchemaTest {

    @Test
    void inferenceEventIsFrozen() {
        EventType type = EventType.getEventType(InferenceEvent.class);
        assertEquals("jinfer.Inference", type.getName());
        assertEquals("Inference", type.getLabel());
        assertEquals(List.of("jinfer", "Inference"), type.getCategoryNames());
        assertEquals(
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
                        "finishReason",
                        "errorType"),
                declared(type),
                "jinfer.Inference fields are public API - adding is compatible, renaming is not");
    }

    @Test
    void modelLoadEventIsFrozen() {
        EventType type = EventType.getEventType(ModelLoadEvent.class);
        assertEquals("jinfer.ModelLoad", type.getName());
        assertEquals(List.of("jinfer", "Lifecycle"), type.getCategoryNames());
        assertEquals(
                Set.of(
                        "model",
                        "architecture",
                        "contextLength",
                        "dimensions",
                        "weightsBytes",
                        "mapped"),
                declared(type));
    }

    @Test
    void runtimeEventIsFrozen() {
        EventType type = EventType.getEventType(RuntimeEvent.class);
        assertEquals("jinfer.Runtime", type.getName());
        assertEquals(List.of("jinfer", "Lifecycle"), type.getCategoryNames());
        assertEquals(Set.of("vectorBits", "decodeThreads"), declared(type));
    }

    @Test
    void speculationAndDecodeAreFrozen() {
        EventType speculation = EventType.getEventType(SpeculationEvent.class);
        assertEquals("jinfer.Speculation", speculation.getName());
        assertEquals(Set.of("draftedTokens", "acceptedTokens", "forwards"), declared(speculation));

        EventType decode = EventType.getEventType(DecodeEvent.class);
        assertEquals("jinfer.Decode", decode.getName());
        assertEquals(
                Set.of(),
                declared(decode),
                "Decode carries no fields: duration and thread ARE the payload, and a logprob"
                        + " here would cost a softmax per token and perturb what it measures");
        assertTrue(
                !decode.isEnabled(),
                "the only event whose frequency scales with output length must default to off");
    }

    /**
     * Oracle's guidelines: headline-style capitalization, no trailing punctuation, and never the
     * word "Event" - labels are what JMC renders as column headers.
     */
    @Test
    void everyLabelFollowsTheJdkConvention() {
        List<Class<? extends Event>> events =
                List.of(
                        InferenceEvent.class,
                        ModelLoadEvent.class,
                        RuntimeEvent.class,
                        SpeculationEvent.class,
                        DecodeEvent.class);
        for (Class<? extends Event> event : events) {
            EventType type = EventType.getEventType(event);
            check(type.getLabel(), type.getName());
            for (ValueDescriptor field : type.getFields()) {
                if (declared(type).contains(field.getName())) {
                    check(field.getLabel(), type.getName() + "#" + field.getName());
                }
            }
        }
    }

    private static void check(String label, String what) {
        assertNotNull(label, what + " has no @Label");
        assertTrue(Character.isUpperCase(label.charAt(0)), what + ": label must start capitalized");
        assertTrue(!label.endsWith("."), what + ": labels take no trailing punctuation");
        assertTrue(!label.contains("Event"), what + ": labels omit the word Event");
    }

    /** Field names minus the ones JFR adds to every event. */
    private static Set<String> declared(EventType type) {
        Set<String> builtIn = Set.of("startTime", "duration", "eventThread", "stackTrace");
        return type.getFields().stream()
                .map(ValueDescriptor::getName)
                .filter(name -> !builtIn.contains(name))
                .collect(Collectors.toSet());
    }
}
