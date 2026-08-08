package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;
import java.util.List;
import java.util.function.DoubleConsumer;

/**
 * A reranker recipe: how one model family turns (query, documents) into scores. Implemented by the
 * port that owns the model card, because the framing and the scoring are two halves of ONE
 * convention - they never leak into a provider integration, which only sees numbers.
 *
 * <p>Two shapes exist today. A {@link CrossEncoder} judges each pair in a causal forward and reads
 * a verdict (Qwen3-Reranker) - it implements three template methods and inherits the
 * frame-once-rewind-per-candidate loop. A late-interaction recipe (LFM2.5-ColBERT) embeds query and
 * documents separately and scores by MaxSim - it owns {@link #scoreAll} outright, because frame
 * reuse is structurally impossible for it: the forward is bidirectional, and short-conv state is
 * not addressable by position.
 */
public interface Reranker<S extends RuntimeState> {

    /** The backbone this recipe scores with - the one it was constructed around. */
    Model<?, ?, S> model();

    /** The task instruction this family's card ships as its default ("" when it has no slot). */
    String defaultInstruction();

    /**
     * Whether this recipe HAS an instruction slot. False for late interaction (MaxSim has no
     * prompt), and checked where an instruction is BOUND - a builder, a bean - so a misconfigured
     * application fails at build, never on its first request.
     */
    default boolean hasInstructionSlot() {
        return true;
    }

    /**
     * Scores every document against {@code query}, one score to {@code sink} per document in input
     * order; returns the exact total token count. {@code state} belongs to this call (reset it) and
     * comes from {@link #model()}; the recipe claims it as it works - through {@code ingest}, or
     * explicitly when it also reads rows between forwards.
     */
    int scoreAll(
            S state, String instruction, String query, List<String> documents, DoubleConsumer sink);

    /**
     * The cross-encoder shape: scaffold + instruction + query framed ONCE, then each candidate
     * ingested after a cursor rewind and judged from its last row - K documents cost {@code |frame|
     * + sum|document|} tokens instead of {@code K * |frame + document|}. Sound because the frame is
     * a strict prefix and rows past the cursor are masked, then overwritten (the law {@link
     * RuntimeState#resumeAt} rests on). Requires a pure-attention port: rewind is a cursor move.
     *
     * <p>Two tokenization domains, as everywhere: scaffolding is emitted as trusted ids; the
     * instruction, query and document go through the plain path and can never mint control tokens.
     */
    interface CrossEncoder<S extends RuntimeState> extends Reranker<S> {

        /** Scaffold + instruction + query, up to and including the document opener. */
        Batch head(String instruction, String query);

        /** The candidate, plus the scaffold that closes the judge turn. */
        Batch document(String document);

        /**
         * The verdict for the pair just ingested (the last retained row): [0,1], higher is more
         * relevant - a {yes, no} softmax, a lone affirmative logit through a sigmoid, an
         * expectation over grade digits; only the port knows which.
         */
        double score(S state);

        @Override
        default int scoreAll(
                S state,
                String instruction,
                String query,
                List<String> documents,
                DoubleConsumer sink) {
            Batch head = head(instruction, query);
            int total = head.count();
            state.reset();
            ingest(state, head);
            int framePositions = state.position();
            for (int i = 0; i < documents.size(); i++) {
                Batch document = document(documents.get(i));
                if (framePositions + document.count() > state.contextCapacity()) {
                    throw new IllegalArgumentException(
                            "document "
                                    + i
                                    + " frames to "
                                    + (framePositions + document.count())
                                    + " tokens, over the "
                                    + state.contextCapacity()
                                    + "-token context - raise contextLength(...) or chunk"
                                    + " smaller");
                }
                total += document.count();
                state.resumeAt(framePositions);
                ingest(state, document);
                sink.accept(score(state));
            }
            return total;
        }

        private void ingest(S state, Batch batch) {
            for (Batch chunk : Batch.prepare(List.of(batch), state.batchCapacity())) {
                model().ingest(state, chunk);
            }
        }
    }
}
