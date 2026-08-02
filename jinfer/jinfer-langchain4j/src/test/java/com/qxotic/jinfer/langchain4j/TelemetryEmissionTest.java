package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;

/**
 * Telemetry that drifts from the API it describes is worse than none, so these assert the event
 * against the same run's own numbers rather than against constants.
 */
class TelemetryEmissionTest {

    private static final String PROMPT = "Name one colour.";

    @Test
    void oneInferenceEventPerCallCarryingThatCallsNumbers() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        Path jfr = Files.createTempFile("jinfer-emission", ".jfr");
        String reply;
        try (var model = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(16).build()) {
            try (Recording recording = new Recording()) {
                recording.enable("jinfer.Inference");
                recording.enable("jinfer.ModelLoad");
                recording.start();
                reply = model.chat(PROMPT);
                recording.stop();
                recording.dump(jfr);
            }
        }

        List<RecordedEvent> inference = eventsOf(jfr, "jinfer.Inference");
        assertEquals(1, inference.size(), "one chat call must emit exactly one event");
        RecordedEvent event = inference.get(0);

        assertEquals("chat", event.getString("operation"));
        assertEquals("text", event.getString("outputType"));
        assertEquals("", event.getString("errorType"), "a successful call reports no error");
        assertNotNull(event.getString("finishReason"));
        assertTrue(event.getString("model").endsWith(".gguf"), event.getString("model"));

        assertTrue(event.getInt("inputTokens") > 0, "the prompt had tokens");
        assertTrue(event.getInt("outputTokens") > 0, "the reply had tokens");
        assertTrue(event.getInt("outputTokens") <= 16, "maxTokens must bound the report");
        assertTrue(event.getLong("prefillTime") > 0, "prefill was measured");
        assertTrue(event.getLong("decodeTime") > 0, "decode was measured");
        assertTrue(
                event.getDuration().toNanos()
                        >= event.getLong("prefillTime") + event.getLong("decodeTime"),
                "the phases are disjoint and both inside the call");
        assertTrue(reply != null && !reply.isBlank());
    }

    /**
     * Observability must not be Heisenbergian. jinfer is deterministic from a warm JVM, so the
     * tokens are identical with recording off and on - an event that perturbed decoding (a stray
     * softmax, an allocation in the loop) would break this.
     */
    @Test
    void recordingDoesNotChangeWhatTheModelProduces() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        try (var model =
                JinferChatModel.builder().modelPath(gguf).maxOutputTokens(24).seed(7L).build()) {
            model.chat(PROMPT); // warm up: interpreted and compiled kernels differ by ~1 LSB

            String withoutJfr = model.chat(PROMPT);
            String withJfr;
            try (Recording recording = new Recording()) {
                recording.enable("jinfer.Inference");
                recording.start();
                withJfr = model.chat(PROMPT);
                recording.stop();
            }
            assertEquals(withoutJfr, withJfr, "recording changed the generated tokens");
        }
    }

    private static List<RecordedEvent> eventsOf(Path jfr, String name) throws Exception {
        try (RecordingFile file = new RecordingFile(jfr)) {
            List<RecordedEvent> found = new java.util.ArrayList<>();
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                if (event.getEventType().getName().equals(name)) found.add(event);
            }
            return found;
        }
    }
}
