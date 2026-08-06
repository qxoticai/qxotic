package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.BaseState;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.Models;
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
 * reranker GGUF (the Qwen3-Reranker family; any reranker port on the classpath loads via the same
 * architecture dispatch as the chat models). Wire it into a {@code RetrievalAugmentationAdvisor}'s
 * post-retrieval stage to reorder what the vector store returned by whether each document actually
 * ANSWERS the query, rather than by embedding similarity.
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
    private final ReentrantLock lock = new ReentrantLock(true); // single-stream, like ChatEngine
    private final String instruction;
    private final int topK;
    private final double minScore;

    private JinferDocumentPostProcessor(Builder b) {
        // ONE arena for weights and state, adopted by the state: state.close() frees everything
        // (idempotent, blocking, Cleaner-backstopped - all BaseState's laws, implemented once)
        Arena arena = Arena.ofShared();
        try {
            try {
                // same contract as the chat and embedding builders: <= 0 means the model's own
                // maximum (-1 to the loader); a literal 0 would crash the port's tensor sizing
                this.loaded = Models.loadReranker(b.modelPath, arena);
            } catch (IOException e) {
                throw new UncheckedIOException("failed to load " + b.modelPath, e);
            }
            this.state = newState(loaded, b.contextLength, arena);
            // the card's own wording is only knowable once the port is loaded
            this.instruction =
                    b.instruction == null ? loaded.reranker().defaultInstruction() : b.instruction;
            this.topK = b.topK;
            this.minScore = b.minScore;
        } catch (RuntimeException | Error e) {
            arena.close(); // a leaked ofShared arena has no Cleaner: free before failing
            throw e;
        }
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
     * Blocking, idempotent: waits out any in-flight call (the fair lock), then frees the single
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
        private Path modelPath;
        private int contextLength;
        private String instruction;
        private int topK;
        private double minScore;

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

        /** Keep only the {@code topK} best documents; 0 (default) keeps all of them. */
        public Builder topK(int topK) {
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
            if (modelPath == null) throw new IllegalArgumentException("modelPath is required");
            return new JinferDocumentPostProcessor(this);
        }
    }
}
