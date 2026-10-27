package com.qxotic.jinfer.models.lfm3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * {@link ModelProvider} service for {@code lfm3}: LFM3, LFM3-VL and LFM3-Omni, the towers from the
 * {@code mmproj} companion. Sampling comes from the GGUF's {@code general.sampling.*}.
 */
public final class Lfm3Provider implements ModelProvider {

    @Override
    public Set<String> architectures() {
        return Set.of("lfm3");
    }

    @Override
    public Map<String, String> companionFiles() {
        return Map.of("media", "mmproj");
    }

    @Override
    public LoadedModel<?> loadLanguage(
            FileChannel fileChannel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Lfm3 model = Lfm3.loadModel(fileChannel, gguf, arena, tokenizer);
        Path media = companions.get("media");
        if (media != null) model = model.withMedia(media, arena);
        Tokenizer tok = model.tokenizer();
        return new LoadedModel<>(
                model,
                tok,
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                SpecialTokens.stops(tok, -1, "<|im_end|>", "<|endoftext|>"),
                Models.modelSeed(fileChannel),
                Optional.of(new Lfm3ChatTemplate(model)),
                LoadedModel.SamplingDefaults.NONE);
    }
}
