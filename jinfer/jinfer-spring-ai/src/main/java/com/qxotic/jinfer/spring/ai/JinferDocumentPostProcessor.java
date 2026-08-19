package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.hub.ModelStore;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.IntStream;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor;

/**
 * Spring AI {@link DocumentPostProcessor} backed by jinfer: in-process CPU reranking over a local
 * reranker GGUF (Qwen3-Reranker's judge, LFM2.5-ColBERT's MaxSim; any reranker port on the
 * classpath loads via the same architecture dispatch as the chat models). Wire it into a {@code
 * RetrievalAugmentationAdvisor}'s post-retrieval stage to reorder what the vector store returned by
 * whether each document actually ANSWERS the query, rather than by embedding similarity.
 *
 * <p>Every document of one call shares the judge frame up to the document slot: it is prefilled
 * ONCE and each candidate re-ingests only its own tokens, so K documents cost {@code |frame| +
 * sum|document|} instead of {@code K * |frame + document|}.
 *
 * <p>The returned documents carry their relevance in {@link Document#getScore()} ([0,1], higher is
 * better) - overwriting the retrieval similarity the store put there - and are sorted best first;
 * ties keep the incoming order. {@code minScore} drops documents outright, {@code topK} truncates.
 *
 * <p>Concurrency contract as everywhere: an instance is ONE serial pipeline (one reusable
 * full-context state, rewound between documents); for parallel pipelines build several instances -
 * weights are shared via the OS page cache. Run with jinfer's JVM flags: {@code --enable-preview
 * --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED}.
 */
public final class JinferDocumentPostProcessor implements DocumentPostProcessor, AutoCloseable {

    private final LoadedReranker<?> loaded;
    private final RuntimeState state; // one reusable state; scoreAll resets it per call
    private final Arena arena;
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine
    private final String instruction;
    private final int topK;
    private final double minScore;
    private final int contextCapacity;
    private final boolean ownsWeights; // false = the caller loaded the model and keeps the arena

    private JinferDocumentPostProcessor(Builder b) {
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
            this.topK = b.topK;
            this.minScore = b.minScore;
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
    public JinferDocumentPostProcessor fork() {
        if (ownsWeights) {
            throw new IllegalStateException(
                    "this model owns its weights and frees them at close - a fork would dangle."
                            + " Load once into YOUR arena instead: Models.loadReranker(path,"
                            + " arena), build with model(loaded), then fork freely");
        }
        Builder b =
                builder()
                        .model(loaded)
                        .contextLength(contextCapacity)
                        .topK(topK)
                        .minScore(minScore);
        if (loaded.reranker().hasInstructionSlot()) b.instruction(instruction);
        return b.build();
    }

    private static <S extends ContextState> S newState(
            LoadedReranker<S> l, int contextCapacity, Arena arena) {
        return l.model().newState(contextCapacity, new PanamaMemoryArena(arena));
    }

    /**
     * Blocking, idempotent: waits out any in-flight call (the fair lock), then frees the single
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
    public List<Document> process(Query query, List<Document> documents) {
        if (documents.isEmpty()) {
            return documents; // a retriever that found nothing must not pay a prefill
        }
        List<String> texts = documents.stream().map(JinferDocumentPostProcessor::text).toList();
        double[] scores = new double[documents.size()];
        int[] next = {0};
        // one serial pipeline per instance (the concurrency contract): concurrent callers queue
        // fairly, exactly like the chat and embedding surfaces
        lock.lock();
        try {
            loaded.scoreAll(state, instruction, query.text(), texts, s -> scores[next[0]++] = s);
        } finally {
            lock.unlock();
        }
        return IntStream.range(0, documents.size())
                .boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> scores[i]).reversed())
                .filter(i -> scores[i] >= minScore)
                .limit(topK <= 0 ? documents.size() : topK)
                .map(i -> documents.get(i).mutate().score(scores[i]).build())
                .toList();
    }

    /** A reranker judges TEXT against a query; a media document has nothing to judge. */
    private static String text(Document document) {
        if (!document.isText()) {
            throw new IllegalArgumentException(
                    "document "
                            + document.getId()
                            + " carries media, not text: a reranker scores (query, text) pairs");
        }
        return document.getText();
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
        private int topK;
        private double minScore;

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
         * <p>You own its weights arena: {@link JinferDocumentPostProcessor#close()} frees only this
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

        /** Keep only the {@code topK} best documents; 0 (default) keeps all of them. */
        public Builder topK(int topK) {
            // 0 is the documented "keep all" sentinel; a negative is a caller bug, not a synonym
            if (topK < 0)
                throw new IllegalArgumentException("topK must be >= 0 (0 keeps all): " + topK);
            this.topK = topK;
            return this;
        }

        /**
         * Drop documents scoring below this; 0 (default) keeps all. The verdict is a probability,
         * so 0.5 reads as "the model would have answered yes" - an off-topic question then empties
         * the context instead of grounding the answer in the least-bad chunk.
         */
        public Builder minScore(double minScore) {
            this.minScore = minScore;
            return this;
        }

        public JinferDocumentPostProcessor build() {
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
            return new JinferDocumentPostProcessor(this);
        }
    }
}
