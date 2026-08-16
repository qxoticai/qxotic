package com.qxotic.jinfer.telemetry;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Pins the bounded string vocabulary used by JFR and OpenTelemetry adapters. */
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
    void factoryDefaultsKeepEveryStringFilterable(@TempDir Path directory) throws Exception {
        Path recordingPath = directory.resolve("vocabulary.jfr");
        try (Recording recording = new Recording()) {
            recording.enable("jinfer.Inference");
            recording.start();

            InferenceEvent ok =
                    InferenceEvent.started(
                            "model.gguf", InferenceEvent.EMBEDDINGS, InferenceEvent.TEXT);
            ok.inputTokens = 7;
            ok.end();
            ok.commit();

            InferenceEvent rejected =
                    InferenceEvent.started("model.gguf", InferenceEvent.CHAT, InferenceEvent.TEXT);
            rejected.errorType = "queue-full";
            rejected.end();
            rejected.commit();

            recording.stop();
            recording.dump(recordingPath);
        }

        List<RecordedEvent> events = new ArrayList<>();
        try (RecordingFile file = new RecordingFile(recordingPath)) {
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
                assertTrue(event.getString(field) != null, field + " must never be null");
            }
            assertTrue(OPERATIONS.contains(event.getString("operation")));
            assertTrue(OUTPUT_TYPES.contains(event.getString("outputType")));
            assertTrue(event.getString("errorType").length() < 40);
        }
    }

    @Test
    void operationNamesMatchOpenTelemetry() {
        assertEquals("chat", InferenceEvent.CHAT);
        assertEquals("embeddings", InferenceEvent.EMBEDDINGS);
        assertEquals("generate_content", InferenceEvent.GENERATE_CONTENT);
        assertEquals("speech", InferenceEvent.SPEECH);
    }
}
