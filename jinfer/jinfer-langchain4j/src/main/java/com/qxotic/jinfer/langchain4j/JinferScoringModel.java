package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.PanamaMemoryArena;
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
    private final Arena arena;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine
    private final String instruction;
    private final int contextCapacity;
    private final boolean ownsWeights; // false = the caller loaded the model and keeps the arena

    private JinferScoringModel(Builder b) {
        // ONE private arena. The state borrows it; close() closes the state first, then the arena.
        // Weights land in it only when THIS instance loads them; caller-loaded weights stay in the
        // caller's arena.
        this.arena = Arenas.newCrossThread();
        try {
            this.ownsWeights = b.loaded == null;
            try {
                this.loaded = b.loaded != null ? b.loaded : Models.loadReranker(b.modelPath, arena);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            int modelContextLength = loaded.model().configuration().contextLength();
            this.contextCapacity =
                    b.contextLength == 0
                            ? modelContextLength
                            : Math.min(b.contextLength, modelContextLength);
            this.state = newState(loaded, contextCapacity, arena);
            // the card's own wording is only knowable once the port is loaded
            if (b.instruction != null && !loaded.reranker().hasInstructionSlot())
                throw new IllegalArgumentException(
                        "this reranker has no instruction slot (late interaction scores by"
                                + " MaxSim) - drop instruction(...)");
            this.instruction =
                    b.instruction == null ? loaded.reranker().defaultInstruction() : b.instruction;
        } catch (RuntimeException | Error e) {
            Arenas.close(arena); // free before failing (best-effort: ofAuto self-manages)
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
        Builder b = builder().model(loaded).contextLength(contextCapacity);
        if (loaded.reranker().hasInstructionSlot()) b.instruction(instruction);
        return b.build();
    }

    private static <S extends ContextState> S newState(
            LoadedReranker<S> l, int contextCapacity, Arena arena) {
        return l.model().newState(contextCapacity, new PanamaMemoryArena(arena));
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
            state.close();
            Arenas.close(arena);
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
        private int contextLength = 2048;
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
         * instance's state, so close your arena after every instance built on it, never before.
         * Getting the order wrong sequentially is caught fail-fast (a safety canary throws
         * IllegalStateException at the next request); freeing the arena DURING a request is a data
         * race and can still crash the VM.
         */
        public Builder model(LoadedReranker<?> loaded) {
            this.source = loaded;
            return this;
        }

        /**
         * Upper bound on the encoded query-and-document context, in tokens. The default is 2048.
         * {@code 0} uses the model's declared context length; otherwise the effective capacity is
         * the smaller of this value and that length.
         *
         * @throws IllegalArgumentException if {@code contextLength < 0}
         */
        public Builder contextLength(int contextLength) {
            if (contextLength < 0)
                throw new IllegalArgumentException(
                        "contextLength must be >= 0 (0 uses the model maximum): " + contextLength);
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
                case String ref -> modelPath = ModelStore.standard().resolve(ref);
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
