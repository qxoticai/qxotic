package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpeculativeDecoding;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Test;

final class ChatEngineTimeoutTest {

    private static final Duration TIMEOUT = Duration.ofMillis(100);
    private static final List<Batch> PROMPT = List.of(Batch.prefill(new int[] {1, 2, 3}));

    @Test
    void deadlineExhaustedByFinalPrefillChunkNeverEntersSpeculativeDecoder() {
        ProbeModel model = new ProbeModel(() -> sleep(Duration.ofMillis(200)));
        AtomicInteger sampled = new AtomicInteger();
        try (ChatEngine engine = engine(model)) {
            ChatEngine.Outcome outcome =
                    engine.generate(PROMPT, sampler(sampled), 1, TIMEOUT, listener());

            assertEquals(1, model.ingests.get(), "the final prefill chunk must complete");
            assertEquals(0, model.speculations.get(), "speculative decoding must not start");
            assertEquals(0, sampled.get(), "plain decoding must not start");
            assertEquals(Generator.FinishReason.TIMEOUT, outcome.result().finishReason());
            assertEquals(0, outcome.result().completionTokens());
            assertEquals(Duration.ZERO, outcome.result().decodeTime());
            assertTrue(outcome.result().promptTime().compareTo(TIMEOUT) >= 0);
        }
    }

    @Test
    void cancellationDuringFinalPrefillChunkNeverEntersPlainDecoder() {
        AtomicBoolean cancelled = new AtomicBoolean();
        ProbeModel model = new ProbeModel(() -> cancelled.set(true));
        AtomicInteger sampled = new AtomicInteger();
        try (ChatEngine engine = engine(model).speculationDepth(0)) {
            ChatEngine.Outcome outcome =
                    engine.generate(
                            PROMPT, sampler(sampled), 1, Duration.ZERO, listener(), cancelled::get);

            assertEquals(1, model.ingests.get(), "the final prefill chunk must complete");
            assertEquals(0, model.speculations.get(), "speculative decoding must not start");
            assertEquals(0, sampled.get(), "plain decoding must not start");
            assertNull(outcome.result(), "cancellation has no generation result");
        }
    }

    @Test
    void ttftIsRecordedAtTheFirstPlainAndSpeculativeToken() throws Exception {
        ProbeModel model = new ProbeModel(() -> {});
        Path recordingPath = Files.createTempFile("jinfer-ttft", ".jfr");
        try (ChatEngine engine = engine(model);
                Recording recording = new Recording()) {
            recording.enable("jinfer.Inference");
            recording.start();

            engine.speculationDepth(0);
            completeOneToken(
                    engine,
                    ignored -> {
                        sleep(Duration.ofMillis(2));
                        return 1;
                    });
            engine.speculationDepth(4);
            completeOneToken(engine, ignored -> 1);

            recording.stop();
            recording.dump(recordingPath);
        }

        List<RecordedEvent> events = events(recordingPath, "jinfer.Inference");
        assertEquals(2, events.size());
        for (RecordedEvent event : events) {
            assertTrue(event.getLong("timeToFirstToken") > 0);
            assertTrue(event.getLong("timeToFirstToken") <= event.getDuration().toNanos());
        }
        assertEquals(1, model.speculations.get(), "the second pass must exercise MTP");
    }

    @Test
    void ttftIsZeroWhenNoTokenIsSampled() throws Exception {
        ProbeModel model = new ProbeModel(() -> {});
        Path recordingPath = Files.createTempFile("jinfer-no-ttft", ".jfr");
        try (ChatEngine engine = engine(model);
                Recording recording = new Recording()) {
            engine.speculationDepth(0);
            recording.enable("jinfer.Inference");
            recording.start();
            try (ChatEngine.Prepared prepared =
                    ChatEngine.Prepared.raw(
                            new int[] {1, 2, 3}, ignored -> 1, 0, Duration.ZERO, List.of())) {
                engine.complete(prepared, ChatEngine.ReplySink.NONE);
            }
            recording.stop();
            recording.dump(recordingPath);
        }

        RecordedEvent event = events(recordingPath, "jinfer.Inference").getFirst();
        assertEquals(0, event.getLong("timeToFirstToken"));
        assertEquals(0, event.getInt("outputTokens"));
    }

    private static void completeOneToken(ChatEngine engine, Sampler sampler) {
        try (ChatEngine.Prepared prepared =
                ChatEngine.Prepared.raw(
                        new int[] {1, 2, 3}, sampler, 1, Duration.ZERO, List.of())) {
            engine.complete(prepared, ChatEngine.ReplySink.NONE);
        }
    }

    private static List<RecordedEvent> events(Path recording, String name) throws Exception {
        try (RecordingFile file = new RecordingFile(recording)) {
            List<RecordedEvent> found = new ArrayList<>();
            while (file.hasMoreEvents()) {
                RecordedEvent event = file.readEvent();
                if (event.getEventType().getName().equals(name)) found.add(event);
            }
            return found;
        }
    }

    private static ChatEngine engine(ProbeModel model) {
        LoadedModel<ProbeState> loaded =
                new LoadedModel<>(
                        model,
                        emptyTokenizer(),
                        "",
                        Set.of(),
                        ContentKey.sha256(new byte[] {1}),
                        Optional.empty(),
                        LoadedModel.SamplingDefaults.NONE);
        PromptCache.Options cache =
                PromptCache.Options.DEFAULTS
                        .withContextCapacity(16)
                        .withBlockBudget(0)
                        .withRetainedSessions(1);
        return new ChatEngine(loaded, "timeout-test", cache);
    }

    private static Sampler sampler(AtomicInteger calls) {
        return ignored -> {
            calls.incrementAndGet();
            return 1;
        };
    }

    private static Generator.GenerationListener listener() {
        return ignored -> true;
    }

    private static void sleep(Duration duration) {
        try {
            Thread.sleep(duration);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
    }

    private static Tokenizer emptyTokenizer() {
        Vocabulary vocabulary =
                (Vocabulary)
                        Proxy.newProxyInstance(
                                Vocabulary.class.getClassLoader(),
                                new Class<?>[] {Vocabulary.class},
                                (proxy, method, args) ->
                                        switch (method.getName()) {
                                            case "size" -> 0;
                                            case "contains" -> false;
                                            case "iterator" -> Collections.emptyIterator();
                                            default ->
                                                    throw new UnsupportedOperationException(
                                                            method.getName());
                                        });
        return (Tokenizer)
                Proxy.newProxyInstance(
                        Tokenizer.class.getClassLoader(),
                        new Class<?>[] {Tokenizer.class},
                        (proxy, method, args) -> {
                            if (method.getName().equals("vocabulary")) return vocabulary;
                            if (method.getName().equals("decodeBytes")) return new byte[] {'x'};
                            throw new UnsupportedOperationException(method.getName());
                        });
    }

    private record Configuration(int vocabularySize, int contextLength)
            implements ContextConfiguration {}

    private static final class ProbeState extends ContextState {
        private ProbeState(
                int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena) {
            super(contextCapacity, batchCapacity, arena, false);
        }

        private void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        @Override
        protected void clearHistory() {}
    }

    private static final class ProbeModel
            implements LanguageModel<Configuration, Void, ProbeState>,
                    SpeculativeDecoding<ProbeState> {

        private static final Configuration CONFIGURATION = new Configuration(32, 64);

        private final Runnable duringIngest;
        private final AtomicInteger ingests = new AtomicInteger();
        private final AtomicInteger speculations = new AtomicInteger();

        private ProbeModel(Runnable duringIngest) {
            this.duringIngest = duringIngest;
        }

        @Override
        public Configuration configuration() {
            return CONFIGURATION;
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public ProbeState newState(
                int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena) {
            return new ProbeState(contextCapacity, batchCapacity, arena);
        }

        @Override
        public ProbeState newState(int contextCapacity, int batchCapacity) {
            return new ProbeState(
                    contextCapacity, batchCapacity, MemoryAllocators.ofArena(Arena.ofAuto()));
        }

        @Override
        public void ingest(ProbeState state, Batch batch) {
            state.exclusively(
                    () -> {
                        ingests.incrementAndGet();
                        duringIngest.run();
                        state.advance(batch);
                    });
        }

        @Override
        public MemoryView<?> logits(ProbeState state, int output) {
            return null;
        }

        @Override
        public boolean speculationReady() {
            return true;
        }

        @Override
        public SpeculationResult speculate(
                ProbeState state,
                Sampler sampler,
                Generator.Constraints constraints,
                int depth,
                Generator.GenerationListener listener,
                SpeculationAudit audit) {
            speculations.incrementAndGet();
            sleep(Duration.ofMillis(2));
            listener.onToken(1);
            return new SpeculationResult(
                    IntSequence.of(1),
                    IntSequence.empty(),
                    OptionalInt.empty(),
                    Generator.FinishReason.LENGTH,
                    Duration.ZERO,
                    0,
                    0,
                    0);
        }
    }
}
