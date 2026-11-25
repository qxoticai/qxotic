package com.qxotic.jinfer.models.inkling2;

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
 * {@link ModelProvider} service for {@code general.architecture} "inkling2_loop": Thinking Machines
 * Inkling 2 Mini, with its vision tower when the {@code mmproj} companion is given. Sampling
 * follows the GGUF's {@code general.sampling.*}.
 */
public final class Inkling2Provider implements ModelProvider {

    @Override
    public Set<String> architectures() {
        return Set.of(Inkling2.ARCHITECTURE);
    }

    @Override
    public Map<String, String> companionFiles() {
        return Map.of("media", "mmproj");
    }

    @Override
    public LoadedModel<?> loadLanguage(
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Inkling2 model = Inkling2.loadModel(channel, gguf, arena, tokenizer);
        Path media = companions.get("media");
        if (media != null) model = model.withMedia(media, arena);
        Tokenizer tok = model.tokenizer();
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        return new LoadedModel<>(
                model,
                tok,
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                SpecialTokens.stops(tok, eos, Inkling2ChatTemplate.END_OF_TURN),
                Models.modelSeed(channel),
                Optional.of(new Inkling2ChatTemplate(tok, model)),
                LoadedModel.SamplingDefaults.NONE);
    }
}
