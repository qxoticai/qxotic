package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;
import java.util.List;
import java.util.function.DoubleConsumer;

/**
 * A reranker port's loaded bundle - the {@link Models#loadReranker} counterpart of {@link
 * LoadedEmbedder}, carrying exactly what a provider integration needs: the backbone and the
 * family's {@link Reranker} recipe (its judge frame and verdict read).
 */
public record LoadedReranker<S extends RuntimeState>(Model<?, ?, S> model, Reranker<S> reranker) {

    public LoadedReranker {
        if (model == null) throw new IllegalArgumentException("null model");
        if (reranker == null) throw new IllegalArgumentException("null reranker");
    }

    /**
     * The provider-integration workhorse: frame the query ONCE, then score each document by
     * rewinding the cursor to the end of the frame and ingesting only the candidate - K documents
     * cost {@code |frame| + sum|document|} tokens instead of {@code K * |frame + document|}. Sound
     * because the frame is a strict prefix and the rows past the cursor are masked, then
     * overwritten (the law {@link RuntimeState#resumeAt} rests on).
     *
     * <p>{@code sink} receives one score per document, in input order. Returns the exact total
     * token count. {@code state} is RESET - it belongs to this call - and must come from {@link
     * #model()}; serialize calls per state (a state is one pipeline). No candidates is not an error
     * and costs nothing: a retriever that found nothing must not pay a prefill.
     */
    public int scoreAll(
            RuntimeState state,
            String instruction,
            String query,
            List<String> documents,
            DoubleConsumer sink) {
        if (documents.isEmpty()) {
            return 0;
        }
        @SuppressWarnings("unchecked") // states of this reranker's model ARE S, by construction
        S s = (S) state;
        Batch head = reranker.head(instruction, query);
        int total = head.count();
        state.reset();
        ingest(s, head);
        int framePositions = state.position();
        for (int i = 0; i < documents.size(); i++) {
            Batch document = reranker.document(documents.get(i));
            if (framePositions + document.count() > state.contextCapacity()) {
                throw new IllegalArgumentException(
                        "document "
                                + i
                                + " frames to "
                                + (framePositions + document.count())
                                + " tokens, over the "
                                + state.contextCapacity()
                                + "-token context - raise contextLength(...) or chunk smaller");
            }
            total += document.count();
            state.resumeAt(framePositions);
            ingest(s, document);
            sink.accept(reranker.score(s));
        }
        return total;
    }

    private void ingest(S state, Batch batch) {
        for (Batch chunk : Batch.prepare(List.of(batch), state.batchCapacity())) {
            model.ingest(state, chunk);
        }
    }
}
