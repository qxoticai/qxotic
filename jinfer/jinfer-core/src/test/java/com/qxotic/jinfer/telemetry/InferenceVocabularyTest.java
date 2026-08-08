package com.qxotic.jinfer.telemetry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;

/**
 * The string fields carry a closed vocabulary, and consumers filter on them - {@code
 * operation=embeddings}, {@code errorType!=""}. Two failure modes make that quietly useless, and
 * both have happened here:
 *
 * <ul>
 *   <li>a null where every other path writes a string, so a filter silently skips those events (a
 *       rejected request reported null {@code outputType} until the {@link InferenceEvent#started}
 *       factory centralised the defaults)
 *   <li>an unbounded value - an exception MESSAGE in {@code errorType} would blow up the constant
 *       pool and make grouping meaningless
 * </ul>
 *
 * <p>Model names are the one deliberately open field: they name a file the user chose.
 */
class InferenceVocabularyTest {

    private static final Set<String> OPERATIONS =
            Set.of(
                    InferenceEvent.CHAT,
                    InferenceEvent.EMBEDDINGS,
                    InferenceEvent.RERANK,
                    InferenceEvent.GENERATE_CONTENT);

    private static final Set<String> OUTPUT_TYPES =
            Set.of(InferenceEvent.TEXT, InferenceEvent.JSON, InferenceEvent.SPEECH);

    @Test
    void theFactoryDefaultsEveryStringSoNoFilterSilentlyMisses() throws Exception {
        Path jfr = Files.createTempFile("jinfer-vocab", ".jfr");
        try (Recording recording = new Recording()) {
            recording.enable("jinfer.Inference");
            recording.start();

            // a plain success, and a failure that never ran - the two shapes every site emits
            InferenceEvent ok =
                    InferenceEvent.started(
                            "m.gguf", InferenceEvent.EMBEDDINGS, InferenceEvent.TEXT);
            ok.inputTokens = 7;
            ok.end();
            ok.commit();

            InferenceEvent rejected =
                    InferenceEvent.started("m.gguf", InferenceEvent.CHAT, InferenceEvent.TEXT);
            rejected.errorType = "queue-full";
            rejected.end();
            rejected.commit();

            recording.stop();
            recording.dump(jfr);
        }

        List<RecordedEvent> events;
        try (RecordingFile file = new RecordingFile(jfr)) {
            events = new ArrayList<>();
            while (file.hasMoreEvents()) events.add(file.readEvent());
        }
        assertEquals(2, events.size());

        for (RecordedEvent event : events) {
            for (String field :
                    List.of(
                            "model",
                            "operation",
                            "outputType",
                            "finishReason",
                            "errorType",
                            "cacheTier")) {
                assertTrue(
                        event.getString(field) != null,
                        field + " must never be null - a filter on it would skip this event");
            }
            assertTrue(
                    OPERATIONS.contains(event.getString("operation")),
                    "unknown operation: " + event.getString("operation"));
            assertTrue(
                    OUTPUT_TYPES.contains(event.getString("outputType")),
                    "unknown outputType: " + event.getString("outputType"));
            assertTrue(
                    event.getString("errorType").length() < 40,
                    "errorType must stay low cardinality - a slug or a class name, never a"
                            + " message");
        }
    }

    /** OpenTelemetry defines these; using our own spelling would break the exporter mapping. */
    @Test
    void operationNamesAreOpenTelemetrysOwn() {
        assertEquals("chat", InferenceEvent.CHAT);
        assertEquals("embeddings", InferenceEvent.EMBEDDINGS);
        assertEquals("generate_content", InferenceEvent.GENERATE_CONTENT);
        assertEquals("speech", InferenceEvent.SPEECH);
    }
}
