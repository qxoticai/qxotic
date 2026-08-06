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
    public java.util.Set<String> architectures() {
        return java.util.Set.of("gemma4");
    }

    @Override
    public LoadedModel<?> load(FileChannel fileChannel, GGUF gguf, int contextLength, Arena arena)
            throws IOException {
        return Gemma4.loadModel(fileChannel, gguf, contextLength, arena).loaded();
    }

    /** Gemma 4's vision and audio encoders ship as a separate mmproj GGUF. */
    @Override
    public java.util.Map<String, String> companionFiles() {
        return java.util.Map.of("media", "mmproj");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            int contextLength,
            Arena arena,
            java.util.Map<String, java.nio.file.Path> companions)
            throws IOException {
        var model = Gemma4.loadModel(fileChannel, gguf, contextLength, arena);
        java.nio.file.Path media = companions.get("media");
        return (media == null ? model : model.withMediaEncoders(media, arena)).loaded();
    }
}
