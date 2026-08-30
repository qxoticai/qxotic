package com.qxotic.jinfer.models.qwen3;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Reranker;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.function.DoubleConsumer;

/**
 * The Qwen3-Reranker recipe: the model card's judge prompt, and its verdict read as {@code P(yes) /
 * (P(yes) + P(no))} from the two verdict tokens at the judged position - one prefill per pair, no
 * sampling, no parsing. The card is the oracle for every string here. (Port of the old tree's
 * Qwen3Reranker onto the MemoryView boundary; its frame-once-rewind-per-candidate loop is
 * retained.)
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
 * <p>The id streams below are hand-split into the exact encode runs the old tree's {@code
 * TokenRuns} contiguity law produces (the two tokenization domains: scaffold TRUSTED, content
 * PLAIN): a special spelling forces a cut and mints its id in place; every stretch between specials
 * is ONE plain encode, and content joins the neighbouring template text. {@code <think>} and {@code
 * </think>} are scaffold too - they are minted as trusted ids, never plain-encoded. The split is
 * pinned token-exact by {@code Qwen3RerankerContractTest} against the shipped scorer's captured
 * ids.
 *
 * <p>Scoring reads two rows of the TIED token-embedding head (reranker GGUFs carry no separate
 * {@code output.weight}) via {@link Qwen3#logit}, so a pair costs two dot products, not a
 * full-vocabulary matmul.
 */
public final class Qwen3Reranker implements Reranker<Qwen3.State> {

    // the trusted scaffold stretches, cut at every special spelling (see the class javadoc)
    private static final String SYSTEM_RUN =
            "system\nJudge whether the Document meets the requirements based on the"
                    + " Query and the Instruct provided. Note that the answer can only be \"yes\""
                    + " or \"no\".";
    private static final String ASSISTANT_RUN = "assistant\n";
    private static final String DEFAULT_INSTRUCTION =
            "Given a web search query, retrieve relevant passages that answer the query";

    private final Qwen3 model;

    @Override
    public Qwen3 model() {
        return model;
    }

    private final int imStart, imEnd, thinkOpen, thinkClose;
    private final int yes, no;

    public Qwen3Reranker(Qwen3 model) {
        this.model = model;
        Tokenizer tokenizer = model.tokenizer();
        this.imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        this.imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        this.thinkOpen = SpecialTokens.require(tokenizer, "<think>");
        this.thinkClose = SpecialTokens.require(tokenizer, "</think>");
        // the whole scoring convention rests on these being single tokens - fail at load
        this.yes = singleToken(tokenizer, "yes");
        this.no = singleToken(tokenizer, "no");
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

    Batch prefix(String instruction, String query) {
        Tokenizer tokenizer = model.tokenizer();
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(imStart);
        ids.addAll(tokenizer.encode(SYSTEM_RUN));
        ids.add(imEnd);
        ids.addAll(tokenizer.encode("\n"));
        ids.add(imStart);
        // the content run joins the trailing template text in ONE plain encode (contiguity law)
        ids.addAll(
                tokenizer.encode(
                        "user\n<Instruct>: "
                                + instruction
                                + "\n<Query>: "
                                + query
                                + "\n<Document>:"));
        return Batch.prefill(ids.build().toArray());
    }

    Batch document(String document) {
        Tokenizer tokenizer = model.tokenizer();
        IntSequence.Builder ids = IntSequence.newBuilder();
        // the leading space belongs to the document run (the seam law, class javadoc)
        ids.addAll(tokenizer.encode(" " + document));
        ids.add(imEnd);
        ids.addAll(tokenizer.encode("\n"));
        ids.add(imStart);
        ids.addAll(tokenizer.encode(ASSISTANT_RUN));
        ids.add(thinkOpen);
        ids.addAll(tokenizer.encode("\n\n"));
        ids.add(thinkClose);
        ids.addAll(tokenizer.encode("\n\n"));
        return Batch.prefill(ids.build().toArray());
    }

    private double score(Qwen3.State state) {
        float yesLogit = model.logit(state, yes);
        float noLogit = model.logit(state, no);
        // softmax over the {yes, no} pair, per the model card
        double max = Math.max(yesLogit, noLogit);
        double pYes = Math.exp(yesLogit - max);
        double pNo = Math.exp(noLogit - max);
        return pYes / (pYes + pNo);
    }

    @Override
    public int scoreAll(
            Qwen3.State state,
            String instruction,
            String query,
            List<String> documents,
            DoubleConsumer sink) {
        return state.exclusively(() -> scoreAll0(state, instruction, query, documents, sink));
    }

    private int scoreAll0(
            Qwen3.State state,
            String instruction,
            String query,
            List<String> documents,
            DoubleConsumer sink) {
        Batch prefix = prefix(instruction, query);
        int total = prefix.count();
        state.reset();
        ingest(state, prefix);
        int prefixLength = state.position();
        for (int i = 0; i < documents.size(); i++) {
            Batch document = document(documents.get(i));
            if (prefixLength + document.count() > state.contextCapacity()) {
                throw new IllegalArgumentException(
                        "document "
                                + i
                                + " frames to "
                                + (prefixLength + document.count())
                                + " tokens, over the "
                                + state.contextCapacity()
                                + "-token context");
            }
            total += document.count();
            state.resumeAt(prefixLength);
            ingest(state, document);
            sink.accept(score(state));
        }
        return total;
    }

    private void ingest(Qwen3.State state, Batch batch) {
        for (Batch chunk : Batch.prepare(List.of(batch), state.batchCapacity())) {
            model.ingest(state, chunk);
        }
    }
}
