package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the NemotronH port's arch-dispatch entry. */
public final class NemotronHProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architectures().contains(architecture);
    }

    @Override
    public java.util.Set<String> architectures() {
        return java.util.Set.of("nemotron_h", "nemotron_h_moe");
    }

    @Override
    public LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena)
            throws IOException {
        return NemotronH.loadModel(fileChannel, gguf, contextLength, arena).loaded();
    }
}
