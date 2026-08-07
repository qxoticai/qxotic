package com.qxotic.jinfer.models.llama;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the Llama port's arch-dispatch entry. */
public final class LlamaProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architectures().contains(architecture);
    }

    @Override
    public java.util.Set<String> architectures() {
        return java.util.Set.of("llama", "minicpm", "mistral3", "smollm3");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            java.util.Map<String, ModelProvider.Companion> companions,
            com.qxotic.toknroll.Tokenizer tokenizer)
            throws IOException {
        return Llama.loadModel(fileChannel, gguf, arena, tokenizer).loaded();
    }
}
