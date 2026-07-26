package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.EmbeddingModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.toknroll.Tokenizer;

/**
 * An embedding port's loaded bundle - the {@link Models#loadEmbedder} counterpart of {@link
 * LoadedModel}, carrying exactly what a provider integration needs: the model, its tokenizer, the
 * port's per-sequence pooling convention ({@code sequenceSuffix} - tokens appended to every encoded
 * sequence, e.g. Qwen3's last-token pooling wants a trailing EOS), and the embedding width (static,
 * so callers never probe with a forward pass).
 */
public record LoadedEmbedder<S extends RuntimeState>(
        EmbeddingModel<?, ?, S> model, Tokenizer tokenizer, int[] sequenceSuffix, int dimension) {

    public LoadedEmbedder {
        if (model == null) throw new IllegalArgumentException("null model");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        if (sequenceSuffix == null) throw new IllegalArgumentException("null sequenceSuffix");
        if (dimension <= 0) throw new IllegalArgumentException("dimension " + dimension);
        sequenceSuffix = sequenceSuffix.clone();
    }
}
