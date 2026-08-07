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

    /** Gemma 4's vision and audio encoders ship as a separate mmproj GGUF. */
    @Override
    public java.util.Map<String, String> companionFiles() {
        return java.util.Map.of("media", "mmproj");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            java.util.Map<String, ModelProvider.Companion> companions,
            com.qxotic.toknroll.Tokenizer tokenizer)
            throws IOException {
        var model = Gemma4.loadModel(fileChannel, gguf, arena, tokenizer);
        ModelProvider.Companion media = companions.get("media");
        return (media == null ? model : model.withMediaEncoders(media, arena)).loaded();
    }
}
