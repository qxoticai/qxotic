package com.qxotic.jinfer.models.nemotronh;

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

/** {@link ModelProvider} service: the NemotronH port's arch-dispatch entry. */
public final class NemotronHProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architectures().contains(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("nemotron_h", "nemotron_h_moe");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        return NemotronH.loadModel(fileChannel, gguf, arena, tokenizer).loaded();
    }
}
