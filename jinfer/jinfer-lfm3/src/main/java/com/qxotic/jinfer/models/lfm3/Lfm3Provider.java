package com.qxotic.jinfer.models.lfm3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.DecisionModel;
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
        String decision = gguf.getStringOrDefault("lfm3.decision.type", "");
        if (!decision.isEmpty()
                || !gguf.getValueOrDefault(boolean.class, "lfm3.attention.causal", true))
            throw new IncompatibleModelException(
                    "'"
                            + path.getFileName()
                            + "' is a decision model"
                            + (decision.isEmpty() ? "" : " (" + decision + ")")
                            + ": it answers typed questions and generates no text, so it is not"
                            + " a language model to chat with");
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

    @Override
    public DecisionModel loadDecisionModel(
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        if (gguf.getStringOrDefault("lfm3.decision.type", "").isEmpty())
            throw new IncompatibleModelException(
                    "'"
                            + path.getFileName()
                            + "' is an LFM3 language model, not a decision model: it was not"
                            + " trained to answer typed questions (Liquid's d1 checkpoints are)");
        Lfm3 model = Lfm3.loadModel(channel, gguf, arena, tokenizer);
        Path media = companions.get("media");
        if (media != null) model = model.withMedia(media, arena);
        return new Lfm3Decision(model, gguf);
    }
}
