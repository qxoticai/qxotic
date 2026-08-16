package com.qxotic.jinfer.x.chat;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContextState;
import com.qxotic.jinfer.x.boundary.EmbeddingModel;
import com.qxotic.jinfer.x.telemetry.InferenceEvent;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * An embedding port's loaded bundle - the {@link Models#loadEmbedder} counterpart of {@link
 * LoadedModel}, carrying exactly what a provider integration needs: the model, its tokenizer, the
 * port's per-sequence framing convention ({@code prefixTokens}/{@code suffixTokens} - tokens
 * wrapped around every encoded sequence: Qwen3's last-token pooling wants a trailing EOS, LFM2.5's
 * CLS pooling reads a leading BOS), the native and minimum supported embedding widths ({@code
 * minimumDimension == dimension} means fixed-width), and the model card's retrieval TEXT framing
 * ({@code queryPrefix}/{@code documentPrefix} - prepended before tokenizing when the caller states
 * a retrieval role; {@code ""} = the card prescribes none for that side). Retrieval-tuned embedders
 * are trained WITH these prefixes - LFM2.5's {@code "query: "}/{@code "document: "} pair, Qwen3's
 * instructed query - and embedding bare text instead silently degrades retrieval quality.
 */
public record LoadedEmbedder<S extends ContextState>(
        EmbeddingModel<?, ?, S> model,
        Tokenizer tokenizer,
        IntSequence prefixTokens,
        IntSequence suffixTokens,
        int dimension,
        int minimumDimension,
        String name,
        String queryPrefix,
        String documentPrefix) {

    public LoadedEmbedder {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(queryPrefix, "queryPrefix");
        Objects.requireNonNull(documentPrefix, "documentPrefix");
        prefixTokens = snapshot(prefixTokens);
        suffixTokens = snapshot(suffixTokens);
        if (dimension <= 0) throw new IllegalArgumentException("dimension " + dimension);
        if (minimumDimension <= 0 || minimumDimension > dimension)
            throw new IllegalArgumentException(
                    "minimumDimension " + minimumDimension + " out of 1.." + dimension);
    }

    private static IntSequence snapshot(IntSequence tokens) {
        return IntSequence.wrap(Objects.requireNonNull(tokens, "tokens").toArray());
    }

    /** Whether this model was trained to preserve quality after prefix truncation. */
    public boolean supportsCustomDimensions() {
        return minimumDimension < dimension;
    }

    private void requireSupportedDimension(int requested) {
        if (requested >= minimumDimension && requested <= dimension) return;
        if (!supportsCustomDimensions()) {
            throw new IllegalArgumentException(
                    name
                            + " has a fixed embedding dimension of "
                            + dimension
                            + "; requested "
                            + requested);
        }
        throw new IllegalArgumentException(
                name
                        + " supports embedding dimensions "
                        + minimumDimension
                        + ".."
                        + dimension
                        + "; requested "
                        + requested);
    }

    /**
     * Copies one reused native embedding row to the heap. Native-width output stays bit-identical;
     * a shorter Matryoshka prefix is L2-normalized as prescribed by the model.
     */
    private float[] copyEmbedding(MemoryView<?> view, int outputDimension) {
        MemoryView<MemorySegment> source = Views.castToSegmentBacked(view, "embedding");
        Views.requireF32(source, "embedding");
        Views.requireContiguous(source, "embedding");
        Views.checkAlive(source, "embedding");
        if (source.shape().size() != dimension)
            throw new IllegalArgumentException(
                    "embedding: expected "
                            + dimension
                            + " values but received "
                            + source.shape().size());
        float[] out = new float[outputDimension];
        Views.copyToArray(source, 0, out, 0, outputDimension, "embedding");
        if (outputDimension == dimension) return out;

        double squaredNorm = 0;
        for (float value : out) squaredNorm += (double) value * value;
        if (squaredNorm == 0) return out;
        float scale = (float) (1 / Math.sqrt(squaredNorm));
        for (int i = 0; i < out.length; i++) out[i] *= scale;
        return out;
    }

    /** Whether the card prescribes retrieval framing at all (either side non-empty). */
    public boolean prefixTrained() {
        return !queryPrefix.isEmpty() || !documentPrefix.isEmpty();
    }

    /** Embeds all texts at the native output width. Each delivered array is caller-owned. */
    public int embedAll(ContextState state, List<String> texts, Consumer<float[]> consumer) {
        return embedAll(state, texts, dimension, consumer);
    }

    /**
     * Encodes, frames and greedily packs all texts, delivering one caller-owned vector per text in
     * order. The consumer runs synchronously on the calling thread after the corresponding model
     * operation has released exclusive state access. Returns the exact framed input-token count.
     */
    public int embedAll(
            ContextState state,
            List<String> texts,
            int outputDimension,
            Consumer<float[]> consumer) {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(texts, "texts");
        Objects.requireNonNull(consumer, "consumer");
        requireSupportedDimension(outputDimension);

        int contextCapacity = state.contextCapacity();
        IntSequence[] sequences = new IntSequence[texts.size()];
        int total = 0;
        for (int i = 0; i < texts.size(); i++) {
            String text = Objects.requireNonNull(texts.get(i), "texts[" + i + "]");
            IntSequence sequence =
                    IntSequence.concatAll(prefixTokens, tokenizer.encode(text), suffixTokens);
            if (sequence.isEmpty()) {
                throw new IllegalArgumentException("text " + i + " produced no tokens");
            }
            if (sequence.length() > contextCapacity) {
                throw new IllegalArgumentException(
                        "text "
                                + i
                                + " is "
                                + sequence.length()
                                + " tokens, over the state's "
                                + contextCapacity
                                + "-token context - use a larger state or chunk smaller");
            }
            sequences[i] = sequence;
            total = Math.addExact(total, sequence.length());
        }
        InferenceEvent event =
                InferenceEvent.started(name, InferenceEvent.EMBEDDINGS, InferenceEvent.TEXT);
        event.inputTokens = total;
        try {
            embedPacked(state, sequences, outputDimension, consumer, event);
            return total;
        } catch (RuntimeException | Error failure) {
            event.errorType = failure.getClass().getSimpleName();
            throw failure;
        } finally {
            event.end();
            event.commit();
        }
    }

    private void embedPacked(
            ContextState state,
            IntSequence[] sequences,
            int outputDimension,
            Consumer<float[]> consumer,
            InferenceEvent event) {
        @SuppressWarnings("unchecked") // states of this embedder's model ARE S, by construction
        S s = (S) state;
        // greedy packing: each group fills the context, one forward pass per group
        int start = 0;
        while (start < sequences.length) {
            int end = start, packed = 0;
            while (end < sequences.length
                    && packed + sequences[end].length() <= state.contextCapacity()) {
                packed += sequences[end].length();
                end++;
            }
            int[] ids = new int[packed];
            int[] len = new int[end - start];
            int at = 0;
            for (int i = start; i < end; i++) {
                IntSequence sequence = sequences[i];
                sequence.copyTo(ids, at);
                at += sequence.length();
                len[i - start] = sequence.length();
            }
            List<float[]> results = new ArrayList<>(len.length);
            long started = System.nanoTime();
            try {
                model.embedAll(
                        s,
                        new Batch.Input.Sequences(new Batch.Input.Tokens(ids), len),
                        view -> results.add(copyEmbedding(view, outputDimension)));
            } finally {
                event.prefillTime += System.nanoTime() - started;
            }
            results.forEach(consumer);
            start = end;
        }
    }
}
