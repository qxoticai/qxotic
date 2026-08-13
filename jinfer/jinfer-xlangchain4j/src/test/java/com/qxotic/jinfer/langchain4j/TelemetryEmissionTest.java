package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.segment.TextSegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
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
        Path gguf =
                TestModels.require(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
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
        Path gguf =
                TestModels.require(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
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

    /** Embeddings are inference too: one event, an honest zero decode, and a batch's tokens. */
    @Test
    void embeddingEmitsAnInferenceEventWithNoDecodePhase() throws Exception {
        Path gguf =
                TestModels.require(
                        "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf");
        Path jfr = Files.createTempFile("jinfer-embed", ".jfr");
        try (var embedder = JinferEmbeddingModel.builder().modelPath(gguf).build()) {
            try (Recording recording = new Recording()) {
                recording.enable("jinfer.Inference");
                recording.start();
                embedder.embedAll(
                        List.of(
                                TextSegment.from("grind the beans"),
                                TextSegment.from("steep the tea")));
                recording.stop();
                recording.dump(jfr);
            }
        }
        List<RecordedEvent> events = eventsOf(jfr, "jinfer.Inference");
        assertEquals(1, events.size(), "one embedAll call, one event");
        RecordedEvent event = events.get(0);
        assertEquals("embeddings", event.getString("operation"));
        assertTrue(
                event.getString("model").endsWith(".gguf"),
                "identity now comes from LoadedEmbedder, not the adapter: "
                        + event.getString("model"));
        assertEquals("", event.getString("errorType"));
        assertTrue(event.getInt("inputTokens") > 0, "the batch had tokens");
        assertEquals(0, event.getInt("outputTokens"), "an encode generates nothing");
        assertEquals(0, event.getLong("decodeTime"), "an encode runs no decode loop");
        assertTrue(event.getLong("prefillTime") > 0);
    }

    /**
     * The per-token event is off by default and must stay that way: it is the only event whose
     * frequency scales with output length.
     */
    @Test
    void decodeIsOffByDefaultAndCountsTokensWhenOn() throws Exception {
        Path gguf =
                TestModels.require(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
        Path off = Files.createTempFile("jinfer-decode-off", ".jfr");
        Path on = Files.createTempFile("jinfer-decode-on", ".jfr");
        try (var model = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(12).build()) {
            model.chat(PROMPT); // warm up before anything is compared

            try (Recording recording = new Recording()) {
                recording.enable("jinfer.Inference"); // Decode NOT enabled
                recording.start();
                model.chat(PROMPT);
                recording.stop();
                recording.dump(off);
            }
            assertEquals(0, eventsOf(off, "jinfer.Decode").size(), "Decode must default to off");

            int outputTokens;
            try (Recording recording = new Recording()) {
                recording.enable("jinfer.Inference");
                recording.enable("jinfer.Decode");
                recording.start();
                model.chat(PROMPT);
                recording.stop();
                recording.dump(on);
            }
            outputTokens = eventsOf(on, "jinfer.Inference").get(0).getInt("outputTokens");
            int decodes = eventsOf(on, "jinfer.Decode").size();
            assertTrue(decodes > 0, "enabling Decode must produce events");
            assertTrue(
                    Math.abs(decodes - outputTokens) <= 1,
                    "one event per decoded token (+/- the stop token): "
                            + decodes
                            + " vs "
                            + outputTokens);
        }
    }

    /** The cache gauge is sampled, so it must appear without any call driving it. */
    @Test
    void promptCacheIsSampledWhileAnEngineIsAlive() throws Exception {
        Path gguf =
                TestModels.require(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
        Path jfr = Files.createTempFile("jinfer-gauge", ".jfr");
        try (var model = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(8).build()) {
            try (Recording recording = new Recording()) {
                recording.enable("jinfer.PromptCache").withPeriod(Duration.ofMillis(200));
                recording.start();
                model.chat(PROMPT);
                Thread.sleep(700); // let the sampler tick
                recording.stop();
                recording.dump(jfr);
            }
        }
        List<RecordedEvent> samples = eventsOf(jfr, "jinfer.PromptCache");
        assertTrue(!samples.isEmpty(), "a live engine's cache must be sampled");
        RecordedEvent sample = samples.get(0);
        assertTrue(sample.getString("model").endsWith(".gguf"));
        assertTrue(sample.getLong("budgetBytes") > 0, "the budget is a real bound");
        assertTrue(sample.getLong("evictions") >= 0, "deltas are never negative");
    }

    /** Every event of {@code name} in the recording - the package's one JFR extraction. */
    static List<RecordedEvent> eventsOf(Path jfr, String name) throws Exception {
        try (RecordingFile file = new RecordingFile(jfr)) {
            List<RecordedEvent> found = new ArrayList<>();
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                if (event.getEventType().getName().equals(name)) found.add(event);
            }
            return found;
        }
    }
}
