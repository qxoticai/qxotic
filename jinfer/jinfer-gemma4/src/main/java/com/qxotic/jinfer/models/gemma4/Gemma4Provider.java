package com.qxotic.jinfer.models.gemma4;

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

/** {@link ModelProvider} service: the Gemma4 port's arch-dispatch entry. */
public final class Gemma4Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "gemma4".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("gemma4");
    }

    /**
     * Gemma 4's companions: the vision/audio encoders (a separate mmproj GGUF) and the MTP draft
     * sidecar that enables self-speculative decoding.
     */
    @Override
    public Map<String, String> companionFiles() {
        return Map.of("media", "mmproj", "speculation", "mtp");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        var model = Gemma4.loadModel(fileChannel, gguf, arena, tokenizer);
        Path media = companions.get("media");
        if (media != null) model.attachMediaEncoders(media, arena);
        Path speculation = companions.get("speculation");
        if (speculation != null) model.attachMtp(speculation, arena);
        return model.loaded();
    }
}
