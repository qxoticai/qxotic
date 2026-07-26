package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.models.qwen35.Qwen3;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.model.scoring.ScoringModel;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * langchain4j {@link ScoringModel} backed by jinfer: in-process CPU reranking over a local
 * causal-LM reranker GGUF (the Qwen3-Reranker convention). Each (query, document) pair is framed
 * with the model card's fixed judge prompt and scored as {@code P(yes) / (P(yes) + P(no))} from the
 * final position's logits - one prefill per pair, no sampling, no parsing.
 *
 * <p>Concurrency contract as everywhere: an instance is ONE serial scoring pipeline (one reusable
 * full-context state, reset between pairs); for parallel pipelines build several instances -
 * weights are shared via the OS page cache.
 */
public final class JinferScoringModel implements ScoringModel {

    // The Qwen3-Reranker prompt frame, verbatim from the model card (the card is the oracle):
    // prefix + "<Instruct>: ..\n<Query>: ..\n<Document>: .." + suffix, then read yes/no logits.
    static final String PREFIX = // package: ScoringBench's naive baseline shares the frame
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the"
                    + " Query and the Instruct provided. Note that the answer can only be \"yes\""
                    + " or \"no\".<|im_end|>\n<|im_start|>user\n";
    static final String SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    static final String DEFAULT_INSTRUCTION =
            "Given a web search query, retrieve relevant passages that answer the query";

    private final Qwen3 model;
    private final Qwen3.State state; // one reusable state; reset() between pairs
    private final java.util.concurrent.locks.ReentrantLock lock =
            new java.util.concurrent.locks.ReentrantLock(true); // single-stream, like ChatEngine
    private final String instruction;
    private final int yes;
    private final int no;

    private JinferScoringModel(Builder b) {
        try {
            this.model = Qwen3.loadModel(b.modelPath, b.contextLength <= 0 ? -1 : b.contextLength);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to load " + b.modelPath, e);
        }
        this.instruction = b.instruction;
        this.state = model.newState(model.config().contextLength(), 512);
        // the whole scoring convention rests on these being single tokens - fail at build
        this.yes = singleToken("yes");
        this.no = singleToken("no");
    }

    private int singleToken(String word) {
        var ids = model.tokenizer().encode(word);
        if (ids.length() != 1) {
            throw new IllegalArgumentException(
                    "not a causal-LM reranker vocabulary: '"
                            + word
                            + "' must be a single token, got "
                            + ids.length());
        }
        return ids.intAt(0);
    }

    @Override
    public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
        List<Double> scores = new ArrayList<>(segments.size());
        // Every pair of this call shares an identical frame up to the document (the card's
        // format deliberately puts the document LAST): prefill it ONCE, then per document
        // rewind the cursor with resumeAt and ingest only (document + suffix). Sound because
        // qwen3 is pure attention - stale KV rows beyond the cursor are masked, the same law
        // the reset gates pin. The seam sits after the ':' so the leading space tokenizes
        // with the document's first word, exactly as the joint encoding would.
        Batch frame =
                new TokenRuns(model.tokenizer())
                        .trusted(PREFIX)
                        .text(
                                "<Instruct>: "
                                        + instruction
                                        + "\n<Query>: "
                                        + query
                                        + "\n<Document>:")
                        .batch();
        int promptTokens = frame.count();
        // one serial scoring pipeline per instance (the concurrency contract): concurrent
        // callers queue fairly, exactly like the chat and embedding surfaces
        lock.lock();
        try {
            state.reset();
            ingest(frame);
            int framePositions = state.position();
            for (TextSegment segment : segments) {
                Batch tail =
                        new TokenRuns(model.tokenizer())
                                .text(" " + segment.text())
                                .trusted(SUFFIX)
                                .batch();
                promptTokens += tail.count();
                state.resumeAt(framePositions);
                ingest(tail);
                scores.add(score());
            }
        } finally {
            lock.unlock();
        }
        return Response.from(scores, new TokenUsage(promptTokens, 0));
    }

    private void ingest(Batch batch) {
        for (Batch chunk : Batch.prepare(List.of(batch), state.batchCapacity())) {
            model.ingest(state, chunk);
        }
    }

    private double score() {
        // exactly two logits via the tied head - no full-vocabulary matmul per pair
        float[] yn = model.logits(state, state.outputCount() - 1, new int[] {yes, no});
        // softmax over the {yes, no} pair, per the model card
        double max = Math.max(yn[0], yn[1]);
        double ey = Math.exp(yn[0] - max);
        double en = Math.exp(yn[1] - max);
        return ey / (ey + en);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path modelPath;
        private int contextLength;
        private String instruction = DEFAULT_INSTRUCTION;

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
         * The task instruction in the judge frame; default is the model card's web-search wording.
         * The card documents task-tuned instructions moving quality 1-5%.
         */
        public Builder instruction(String instruction) {
            this.instruction = instruction;
            return this;
        }

        public JinferScoringModel build() {
            return new JinferScoringModel(this);
        }
    }
}
