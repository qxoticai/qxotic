package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.BaseState;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.hub.ModelStore;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.model.scoring.ScoringModel;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * langchain4j {@link ScoringModel} backed by jinfer: in-process CPU reranking over a local reranker
 * GGUF (Qwen3-Reranker's judge, LFM2.5-ColBERT's MaxSim; any reranker port on the classpath loads
 * via the same architecture dispatch as the chat models). The family's judge prompt and verdict
 * read live in its port - this class maps types and owns the pipeline.
 *
 * <p>Every candidate of one call shares the frame up to the document (the card's format puts the
 * document LAST): it is prefilled ONCE and each document re-ingests only its own tokens, so K
 * documents cost {@code |frame| + sum|document|} instead of {@code K * |frame + document|}.
 *
 * <p>Concurrency contract as everywhere: an instance is ONE serial scoring pipeline (one reusable
 * full-context state, rewound between pairs); for parallel pipelines build several instances -
 * weights are shared via the OS page cache.
 */
public final class JinferScoringModel implements ScoringModel, AutoCloseable {

    private final LoadedReranker<?> loaded;
    private final RuntimeState state; // one reusable state; scoreAll resets it per call
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine
    private final String instruction;
    private final int contextLength; // the builder's raw knob, carried for fork()
    private final boolean ownsWeights; // false = the caller loaded the model and keeps the arena

    private JinferScoringModel(Builder b) {
        // ONE arena adopted by the state: state.close() frees everything this instance allocated
        // (idempotent, blocking, Cleaner-backstopped - all BaseState's laws, implemented once).
        // Weights land in it only when THIS instance loads them; a caller-loaded model stays in
        // the caller's arena, and this arena holds the state alone.
        Arena arena = Arena.ofShared();
        try {
            this.ownsWeights = b.loaded == null;
            try {
                this.loaded = b.loaded != null ? b.loaded : Models.loadReranker(b.modelPath, arena);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.contextLength = b.contextLength;
            this.state = newState(loaded, b.contextLength, arena);
            // the card's own wording is only knowable once the port is loaded
            if (b.instruction != null && !loaded.reranker().hasInstructionSlot())
                throw new IllegalArgumentException(
                        "this reranker has no instruction slot (late interaction scores by"
                                + " MaxSim) - drop instruction(...)");
            this.instruction =
                    b.instruction == null ? loaded.reranker().defaultInstruction() : b.instruction;
        } catch (RuntimeException | Error e) {
            arena.close(); // a leaked ofShared arena has no Cleaner: free before failing
            throw e;
        }
    }

    /**
     * A parallel pipeline over the same weights: fresh state, own lock, own lifecycle. Only a model
     * whose weights YOU loaded can fork - the weights' lifetime is your arena's, so a fork can
     * never dangle. A model that loaded its own weights refuses: it frees them at {@link #close()},
     * and a fork would outlive them.
     */
    public JinferScoringModel fork() {
        if (ownsWeights) {
            throw new IllegalStateException(
                    "this model owns its weights and frees them at close - a fork would dangle."
                            + " Load once into YOUR arena instead: Models.loadReranker(path,"
                            + " arena), build with model(loaded), then fork freely");
        }
        Builder b = builder().model(loaded).contextLength(contextLength);
        if (loaded.reranker().hasInstructionSlot()) b.instruction(instruction);
        return b.build();
    }

    private static <S extends RuntimeState> S newState(
            LoadedReranker<S> l, int contextLength, Arena arena) {
        return l.model()
                .newState(
                        Math.min(
                                contextLength <= 0 ? Integer.MAX_VALUE : contextLength,
                                l.model().config().contextLength()),
                        RuntimeFlags.BATCH_CAPACITY,
                        arena,
                        true);
    }

    /**
     * Blocking, idempotent: waits out any in-flight scoring (the fair lock), then frees the single
     * arena holding weights and state deterministically; later calls fail with
     * IllegalStateException.
     */
    @Override
    public void close() {
        lock.lock();
        try {
            ((BaseState) state).close();
        } finally {
            lock.unlock();
        }
    }

    @Override
    public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
        List<String> documents = segments.stream().map(TextSegment::text).toList();
        List<Double> scores = new ArrayList<>(segments.size());
        int promptTokens;
        // one serial scoring pipeline per instance (the concurrency contract): concurrent
        // callers queue fairly, exactly like the chat and embedding surfaces
        lock.lock();
        try {
            promptTokens = loaded.scoreAll(state, instruction, query, documents, scores::add);
        } finally {
            lock.unlock();
        }
        return Response.from(scores, new TokenUsage(promptTokens, 0));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Object source; // Path | ref/URL String | LoadedReranker: the last setter wins
        private Path modelPath; // derived from source at build()
        private LoadedReranker<?> loaded; // derived from source at build()
        private int contextLength;
        private String instruction;

        /** The reranker GGUF to load. Required. */
        public Builder modelPath(Path modelPath) {
            this.source = modelPath;
            return this;
        }

        /**
         * The model as ONE string: a local GGUF path, a hub ref ({@code hf.co/owner/repo:Q4_K_M})
         * or a pasted browser URL - resolved by {@link #build()} with the rest of the load, so a
         * remote ref downloads there (see the package doc) and the chain never blocks.
         */
        public Builder model(String pathOrRef) {
            this.source = pathOrRef;
            return this;
        }

        /**
         * A model you loaded yourself ({@code Models.loadReranker(path, arena)}) - the
         * weight-sharing seam: several instances (and their {@link #fork() forks}) over ONE loaded
         * copy are parallel pipelines for the price of one load.
         *
         * <p>You own its weights arena: {@link JinferScoringModel#close()} frees only this
         * instance's state, so close your arena after every instance built on it, never before -
         * the tensor hot path reads raw addresses, so a closed weights arena under a live instance
         * is a VM crash, not a catchable exception.
         */
        public Builder model(LoadedReranker<?> loaded) {
            this.source = loaded;
            return this;
        }

        /** Context window; 0 = the model's own maximum. Bounds query+document length. */
        public Builder contextLength(int contextLength) {
            this.contextLength = contextLength;
            return this;
        }

        /**
         * The task instruction in the judge frame; default is the model card's own wording. The
         * cards document task-tuned instructions moving quality 1-5%.
         */
        public Builder instruction(String instruction) {
            this.instruction = instruction;
            return this;
        }

        public JinferScoringModel build() {
            modelPath = null;
            loaded = null;
            switch (source) {
                case String ref -> modelPath = ModelStore.resolve(ref);
                case Path path -> modelPath = path;
                case LoadedReranker<?> l -> loaded = l;
                case null, default ->
                        throw new IllegalArgumentException(
                                "a model is required: model(\"hf.co/owner/repo:Q4_K_M\"),"
                                        + " modelPath(...) or model(LoadedReranker)");
            }
            return new JinferScoringModel(this);
        }
    }
}
