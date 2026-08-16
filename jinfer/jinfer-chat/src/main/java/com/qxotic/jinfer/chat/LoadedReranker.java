package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.boundary.ContextModel;
import com.qxotic.jinfer.boundary.ContextState;
import com.qxotic.jinfer.boundary.Reranker;
import com.qxotic.jinfer.boundary.RuntimeState;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import java.util.List;
import java.util.function.DoubleConsumer;

/**
 * A reranker port's loaded bundle - the {@link Models#loadReranker} counterpart of {@link
 * LoadedEmbedder}, carrying exactly what a provider integration needs: the backbone and the
 * family's {@link Reranker} recipe (its judge frame and verdict read).
 */
public record LoadedReranker<S extends ContextState>(
        ContextModel<?, ?, S> model, Reranker<S> reranker, String name) {

    public LoadedReranker {
        if (model == null) throw new IllegalArgumentException("null model");
        if (reranker == null) throw new IllegalArgumentException("null reranker");
        if (name == null) throw new IllegalArgumentException("null name");
    }

    /**
     * The provider-integration workhorse: delegates to the recipe's {@link Reranker#scoreAll} - the
     * cross-encoder template or a late-interaction loop, whichever the family is - and owns only
     * what is recipe-independent: telemetry, and the empty-candidates shortcut.
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
            return 0; // no prefill, no work, and nothing worth an event
        }
        @SuppressWarnings("unchecked") // states of this reranker's model ARE S, by construction
        S s = (S) state;
        InferenceEvent event =
                InferenceEvent.started(name, InferenceEvent.RERANK, InferenceEvent.TEXT);
        long startNanos = System.nanoTime();
        try {
            int tokens = reranker.scoreAll(s, instruction, query, documents, sink);
            event.inputTokens = tokens;
            return tokens;
        } catch (RuntimeException | Error failure) {
            event.errorType = failure.getClass().getSimpleName();
            throw failure;
        } finally {
            // scoring is all prefill: a reranker emits no tokens
            event.prefillTime = System.nanoTime() - startNanos;
            event.end();
            event.commit();
        }
    }
}
