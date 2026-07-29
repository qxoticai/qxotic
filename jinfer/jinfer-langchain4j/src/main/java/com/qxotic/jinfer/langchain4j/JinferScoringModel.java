package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.BaseState;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.Models;
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
 * GGUF (the Qwen3-Reranker family; any reranker port on the classpath loads via the same
 * architecture dispatch as the chat models). The family's judge prompt and verdict read live in its
 * port - this class maps types and owns the pipeline.
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

    private JinferScoringModel(Builder b) {
        // ONE arena for weights and state, adopted by the state: state.close() frees everything
        // (idempotent, blocking, Cleaner-backstopped - all BaseState's laws, implemented once)
        Arena arena = Arena.ofShared();
        try {
            try {
                // same contract as the chat builders: <= 0 means the model's own maximum (-1 to the
                // loader); a literal 0 would crash the port's tensor sizing
                this.loaded =
                        Models.loadReranker(
                                b.modelPath, b.contextLength <= 0 ? -1 : b.contextLength, arena);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.state = newState(loaded, arena);
            // the card's own wording is only knowable once the port is loaded
            this.instruction =
                    b.instruction == null ? loaded.reranker().defaultInstruction() : b.instruction;
        } catch (RuntimeException | Error e) {
            arena.close(); // a leaked ofShared arena has no Cleaner: free before failing
            throw e;
        }
    }

    private static <S extends RuntimeState> S newState(LoadedReranker<S> l, Arena arena) {
        return l.model()
                .newState(
                        l.model().config().contextLength(),
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
        private Path modelPath;
        private int contextLength;
        private String instruction;

        /** The reranker GGUF to load. Required. */
        public Builder modelPath(Path modelPath) {
            this.modelPath = modelPath;
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
            if (modelPath == null) throw new IllegalArgumentException("modelPath is required");
            return new JinferScoringModel(this);
        }
    }
}
