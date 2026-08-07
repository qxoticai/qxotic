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

    /**
     * Gemma 4's companions: the vision/audio encoders (a separate mmproj GGUF) and the MTP draft
     * sidecar that enables self-speculative decoding.
     */
    @Override
    public java.util.Map<String, String> companionFiles() {
        return java.util.Map.of("media", "mmproj", "speculation", "mtp");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            java.util.Map<String, java.nio.file.Path> companions,
            com.qxotic.toknroll.Tokenizer tokenizer)
            throws IOException {
        var model = Gemma4.loadModel(fileChannel, gguf, arena, tokenizer);
        java.nio.file.Path media = companions.get("media");
        if (media != null) model.attachMediaEncoders(media, arena);
        java.nio.file.Path speculation = companions.get("speculation");
        if (speculation != null) model.attachMtp(speculation, arena);
        return model.loaded();
    }
}
