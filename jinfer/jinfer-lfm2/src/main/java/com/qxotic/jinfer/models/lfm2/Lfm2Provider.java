package com.qxotic.jinfer.models.lfm2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the Lfm2 port's arch-dispatch entry. */
public final class Lfm2Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architecture.startsWith("lfm");
    }

    @Override
    public java.util.Set<String> architectures() {
        return java.util.Set.of("lfm2", "lfm2moe"); // representative: supports() matches lfm*
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            java.util.Map<String, java.nio.file.Path> companions,
            com.qxotic.toknroll.Tokenizer tokenizer)
            throws IOException {
        Lfm2 m = Lfm2.loadModel(fileChannel, gguf, arena, tokenizer);
        // a retrieval checkpoint "loads" as a chat model and then generates noise - refuse by name
        if (!m.config().causalAttention())
            throw new IllegalArgumentException(
                    "this LFM2 checkpoint is a RETRIEVAL model (non-causal attention), not a"
                            + " generative one - load it with Models.loadEmbedder"
                            + " (LFM2.5-Embedding) or Models.loadReranker (LFM2.5-ColBERT)");
        return m.loaded();
    }

    /**
     * The family's reranker is LFM2.5-ColBERT: late interaction (MaxSim over per-token {@code
     * dense_2} embeddings), not a cross-encoder judge. Detected by its own metadata, so a wrong
     * file refuses by name instead of producing numbers.
     */
    @Override
    public com.qxotic.jinfer.chat.LoadedReranker<?> loadReranker(
            FileChannel fileChannel, GGUF gguf, java.nio.file.Path path, Arena arena)
            throws IOException {
        Lfm2 m = Lfm2.loadModel(fileChannel, gguf, arena);
        if (m.config().causalAttention()
                || m.config().embeddingLengthOut() <= 0
                || m.weights().dense2() == null)
            throw new IllegalArgumentException(
                    path.getFileName()
                            + " is not the family's reranker - LFM2 reranking is"
                            + " LFM2.5-ColBERT-350M-GGUF (per-token dense_2 embeddings, MaxSim);"
                            + " a generative checkpoint chats via Models.load, the embedding"
                            + " checkpoint embeds via Models.loadEmbedder");
        int bos = gguf.getValue(int.class, "tokenizer.ggml.bos_token_id");
        int pad = gguf.getValue(int.class, "tokenizer.ggml.padding_token_id");
        return new com.qxotic.jinfer.chat.LoadedReranker<>(
                m, new Lfm2Colbert(m, bos, pad), path.getFileName().toString());
    }

    /**
     * The LFM2.5-Embedding checkpoints: same architecture string as the generative models, told
     * apart by their OWN metadata (non-causal attention + CLS pooling) - so unlike qwen3, a
     * generative GGUF handed here refuses loudly instead of producing numbers.
     */
    @Override
    public com.qxotic.jinfer.chat.LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, java.nio.file.Path path, Arena arena)
            throws IOException {
        Lfm2 m = Lfm2.loadModel(fileChannel, gguf, arena);
        if (m.config().causalAttention() || m.config().poolingType() != Lfm2.POOLING_CLS)
            throw new IllegalArgumentException(
                    path.getFileName()
                            + " is a generative LFM2 checkpoint, not an embedder (embedders"
                            + " declare pooling_type and non-causal attention) - chat with it via"
                            + " Models.load, or embed with LFM2.5-Embedding-350M-GGUF");
        // CLS pooling reads the BOS row, so every sequence leads with it (add_bos in the GGUF)
        int bos = gguf.getValue(int.class, "tokenizer.ggml.bos_token_id");
        return new com.qxotic.jinfer.chat.LoadedEmbedder<Lfm2.State>(
                m,
                m.tokenizer(),
                new int[] {bos},
                new int[0],
                m.config().embeddingLength(),
                path.getFileName().toString());
    }
}
