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
        return Lfm2.loadModel(fileChannel, gguf, arena, tokenizer).loaded();
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
