package com.qxotic.jinfer.models.qwen35;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.llm.SpecialTokens;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;

/**
 * {@link ModelProvider} service: the Qwen port's arch-dispatch entry - Qwen 3.5 (dense + MoE)
 * generative models and the Qwen3-Embedding family ({@code general.architecture} "qwen3").
 */
public final class Qwen35Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "qwen35".equals(architecture)
                || "qwen35moe".equals(architecture)
                || "qwen3".equals(architecture);
    }

    @Override
    public LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena)
            throws IOException {
        if ("qwen3".equals(gguf.getString("general.architecture"))) {
            throw new UnsupportedOperationException(
                    "'qwen3' is the Qwen3-Embedding architecture - load it with"
                            + " Models.loadEmbedder");
        }
        return Qwen35.loadModel(fileChannel, gguf, contextLength, true, arena).loaded();
    }

    @Override
    public LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena) throws IOException {
        if (!"qwen3".equals(gguf.getString("general.architecture"))) {
            return ModelProvider.super.loadEmbedder(fileChannel, gguf, contextLength, arena);
        }
        Qwen3 m = Qwen3.loadModel(fileChannel, gguf, contextLength, true, arena);
        // last-token pooling wants a trailing EOS on every sequence (the llama.cpp convention)
        int eos =
                SpecialTokens.find(m.tokenizer(), "<|endoftext|>")
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "qwen3 vocab has no <|endoftext|>"));
        return new LoadedEmbedder<>(
                m, m.tokenizer(), new int[] {eos}, m.config().embeddingLength());
    }
}
