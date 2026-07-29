package com.qxotic.jinfer.models.qwen3;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Reranker;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;

/**
 * The Qwen3-Reranker recipe: the model card's judge prompt, and its verdict read as {@code P(yes) /
 * (P(yes) + P(no))} from the two verdict tokens at the judged position - one prefill per pair, no
 * sampling, no parsing. The card is the oracle for every string here.
 *
 * <p>Layout: {@code <|im_start|>system\n...only "yes" or "no".<|im_end|>\n<|im_start|>user\n
 * <Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n
 * <|im_start|>assistant\n<think>\n\n</think>\n\n} - the document LAST, so everything before it is a
 * reusable prefix across candidates.
 *
 * <p>The head/document seam sits after the {@code :} of the document opener: the leading space
 * belongs to the document run, so it tokenizes with the document's first word exactly as the joint
 * encoding would - the split is token-identical to one continuous frame.
 *
 * <p>Scoring reads two rows of the TIED token-embedding head (reranker GGUFs carry no separate
 * {@code output.weight}), so a pair costs two dot products, not a full-vocabulary matmul.
 */
final class Qwen3Reranker implements Reranker<Qwen3.State> {

    private static final String PREFIX =
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the"
                    + " Query and the Instruct provided. Note that the answer can only be \"yes\""
                    + " or \"no\".<|im_end|>\n<|im_start|>user\n";
    private static final String SUFFIX =
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    private static final String DEFAULT_INSTRUCTION =
            "Given a web search query, retrieve relevant passages that answer the query";

    private final Qwen3 model;
    private final TokenRuns runs; // prototype: fresh() per frame, one compiled spelling table
    private final int yes;
    private final int no;

    Qwen3Reranker(Qwen3 model) {
        this.model = model;
        this.runs = new TokenRuns(model.tokenizer());
        // the whole scoring convention rests on these being single tokens - fail at load
        this.yes = singleToken(model.tokenizer(), "yes");
        this.no = singleToken(model.tokenizer(), "no");
    }

    private static int singleToken(Tokenizer tokenizer, String word) {
        IntSequence ids = tokenizer.encode(word);
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
    public String defaultInstruction() {
        return DEFAULT_INSTRUCTION;
    }

    @Override
    public Batch head(String instruction, String query) {
        return runs.fresh()
                .trusted(PREFIX)
                .text("<Instruct>: " + instruction + "\n<Query>: " + query + "\n<Document>:")
                .batch();
    }

    @Override
    public Batch document(String document) {
        return runs.fresh().text(" " + document).trusted(SUFFIX).batch();
    }

    @Override
    public double score(Qwen3.State state) {
        float yesLogit = model.logit(state, yes);
        float noLogit = model.logit(state, no);
        // softmax over the {yes, no} pair, per the model card
        double max = Math.max(yesLogit, noLogit);
        double pYes = Math.exp(yesLogit - max);
        double pNo = Math.exp(noLogit - max);
        return pYes / (pYes + pNo);
    }
}
