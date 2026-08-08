package com.qxotic.jinfer.models.llama;

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

/** {@link ModelProvider} service: the Llama port's arch-dispatch entry. */
public final class LlamaProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architectures().contains(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("llama", "minicpm", "mistral3", "smollm3");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        return Llama.loadModel(fileChannel, gguf, arena, tokenizer).loaded();
    }
}
