package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;

/** {@link ModelProvider} service: the Gemma4 port's arch-dispatch entry. */
public final class Gemma4Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "gemma4".equals(architecture);
    }

    @Override
    public LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena)
            throws IOException {
        return Gemma4.loadModel(fileChannel, gguf, contextLength, arena).loaded();
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            int contextLength,
            java.nio.file.Path mediaProjector,
            Arena arena)
            throws IOException {
        return Gemma4.loadModel(fileChannel, gguf, contextLength, arena)
                .withMediaEncoders(mediaProjector, arena)
                .loaded();
    }
}
