package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.TokenRuns;
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
    private static final String PREFIX =
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the"
                    + " Query and the Instruct provided. Note that the answer can only be \"yes\""
                    + " or \"no\".<|im_end|>\n<|im_start|>user\n";
    private static final String SUFFIX =
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    private static final String DEFAULT_INSTRUCTION =
            "Given a web search query, retrieve relevant passages that answer the query";

    private final LoadedModel<?> loaded;
    private final RuntimeState state; // one reusable state; reset() between pairs
    private final String instruction;
    private final int yes;
    private final int no;

    private JinferScoringModel(Builder b) {
        try {
            this.loaded = Models.load(b.modelPath, b.contextLength <= 0 ? -1 : b.contextLength);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to load " + b.modelPath, e);
        }
        this.instruction = b.instruction;
        this.state = loaded.model().newState(loaded.model().config().contextLength(), 512);
        // the whole scoring convention rests on these being single tokens - fail at build
        this.yes = singleToken("yes");
        this.no = singleToken("no");
    }

    private int singleToken(String word) {
        var ids = loaded.tokenizer().encode(word);
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
        int promptTokens = 0;
        for (TextSegment segment : segments) {
            Batch prompt =
                    new TokenRuns(loaded.tokenizer())
                            .trusted(PREFIX)
                            .text(
                                    "<Instruct>: "
                                            + instruction
                                            + "\n<Query>: "
                                            + query
                                            + "\n<Document>: "
                                            + segment.text())
                            .trusted(SUFFIX)
                            .batch();
            promptTokens += prompt.count();
            scores.add(score(loaded.model(), prompt));
        }
        return Response.from(scores, new TokenUsage(promptTokens, 0));
    }

    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> double score(LanguageModel<?, ?, S> model, Batch prompt) {
        S s = (S) state;
        s.reset();
        for (Batch chunk : Batch.prepare(List.of(prompt), s.batchCapacity())) {
            model.ingest(s, chunk);
        }
        FloatTensor logits = model.logits(s, s.outputCount() - 1);
        // log-softmax over the {yes, no} pair, per the model card: exp(yes) / (exp(yes)+exp(no))
        double ly = logits.getFloat(yes);
        double ln = logits.getFloat(no);
        double max = Math.max(ly, ln);
        double ey = Math.exp(ly - max);
        double en = Math.exp(ln - max);
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
