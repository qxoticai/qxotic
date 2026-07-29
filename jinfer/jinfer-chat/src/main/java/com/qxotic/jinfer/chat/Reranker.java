package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.RuntimeState;

/**
 * A reranker recipe: how one model family frames a (query, document) pair into tokens and reads the
 * verdict back out. Implemented by the port that owns the model card - the frame and the verdict
 * are two halves of ONE convention (the judge prompt says "answer yes or no", the verdict reads
 * exactly those two tokens), so they live together in the port and never leak into a provider
 * integration.
 *
 * <p>The document goes LAST by construction. That is what lets {@link LoadedReranker} prefill the
 * frame once per query and re-ingest only the candidate; a family that framed the document before
 * the query would forfeit the reuse, and a bidirectional cross-encoder forfeits it structurally
 * (every token's state depends on the document).
 *
 * <p>Two tokenization domains, as everywhere: scaffolding is emitted as trusted ids, the
 * instruction, query and document go through the plain path and can never mint control tokens.
 *
 * <p>The verdict is one number however the family produces it - a {yes, no} softmax, a lone
 * affirmative logit through a sigmoid, an expectation over grade digits, or a real classification
 * head. Only the port knows which; callers see a score.
 *
 * <p>Requires a state whose rewind is a cursor move, i.e. a pure-attention port: {@link
 * LoadedReranker#scoreAll} rewinds to the frame between candidates, and short-conv or SSM layers
 * carry state that is not addressable by position (those need a checkpoint restore instead).
 */
public interface Reranker<S extends RuntimeState> {

    /** The task instruction this family's card ships as its default. */
    String defaultInstruction();

    /** Scaffold + instruction + query, up to and including the document opener. */
    Batch head(String instruction, String query);

    /** The candidate, plus the scaffold that closes the judge turn. */
    Batch document(String document);

    /**
     * The verdict for the pair just ingested (the last retained row): [0,1], higher is more
     * relevant. Only the last row is addressable because {@link LoadedReranker#scoreAll} ingests
     * one pair at a time; scoring several packed pairs in one forward pass would want an indexed
     * variant, and can add one then.
     */
    double score(S state);
}
