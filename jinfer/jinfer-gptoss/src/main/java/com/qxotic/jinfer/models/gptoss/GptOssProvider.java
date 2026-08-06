package com.qxotic.jinfer.models.gptoss;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the GptOss port's arch-dispatch entry. */
public final class GptOssProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "gpt-oss".equals(architecture);
    }

    @Override
    public java.util.Set<String> architectures() {
        return java.util.Set.of("gpt-oss");
    }

    @Override
    public LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, Arena arena) throws IOException {
        return GptOss.loadModel(fileChannel, gguf, arena).loaded();
    }
}
