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
    public LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, Arena arena) throws IOException {
        return Lfm2.loadModel(fileChannel, gguf, arena).loaded();
    }
}
