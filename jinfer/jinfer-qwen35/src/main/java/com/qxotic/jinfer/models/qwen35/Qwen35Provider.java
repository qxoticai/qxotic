package com.qxotic.jinfer.models.qwen35;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * {@link ModelProvider} service: the Qwen 3.5 generative port's arch-dispatch entry, dense and MoE.
 * The retrieval family (architecture "qwen3": Qwen3-Embedding, Qwen3-Reranker) is a different
 * backbone and lives in jinfer-qwen3.
 */
public final class Qwen35Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architectures().contains(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("qwen35", "qwen35moe");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        return Qwen35.loadModel(fileChannel, gguf, arena, tokenizer).loaded();
    }
}
