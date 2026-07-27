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
        return java.util.Set.of("llama", "minicpm", "mistral3", "smollm3").contains(architecture);
    }

    @Override
    public LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena)
            throws IOException {
        return Llama.loadModel(fileChannel, gguf, contextLength, arena).loaded();
    }
}
