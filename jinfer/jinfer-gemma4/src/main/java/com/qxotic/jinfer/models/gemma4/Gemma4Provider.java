package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** {@link ModelProvider} service: the Gemma 4 port's arch-dispatch entry. */
public final class Gemma4Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "gemma4".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("gemma4");
    }

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
        Gemma4 model = Gemma4.loadModel(fileChannel, gguf, arena, tokenizer);
        Path media = companions.get("media");
        if (media != null) {
            model = model.withMedia(media, arena);
        }
        Path speculation = companions.get("speculation");
        if (speculation != null) {
            model = model.attachMtp(speculation, arena);
        }
        Tokenizer tok = model.tokenizer();
        String source = gguf.getStringOrDefault("tokenizer.chat_template", "");
        // insertion-ordered: the FIRST present name is the id the engine's endTurn emits (the
        // model's own end-of-turn, never the handoff marker). Raw vocabulary lookups, not
        // SpecialTokens.find: "<turn|>" is not flagged special in every checkpoint.
        Set<Integer> stops = new LinkedHashSet<>();
        // <|tool_response> is the HANDOFF: results are runtime-provided by definition, so a
        // model-emitted response marker always means "stop, my call awaits its result"
        for (String name :
                new String[] {
                    "<turn|>", "<end_of_turn>", "<eos>", "<|endoftext|>", "<|tool_response>"
                }) {
            if (tok.vocabulary().contains(name)) {
                stops.add(tok.vocabulary().id(name));
            }
        }
        // "{%- if not enable_thinking -%}" in the add_generation_prompt tail: 12B and 26B carry
        // that branch, E2B does not
        boolean scaffoldsNonThinking = source.contains("not enable_thinking");
        return new LoadedModel<>(
                model,
                tok,
                source,
                stops,
                Models.modelSeed(fileChannel),
                Optional.of(new Gemma4ChatTemplate(model, scaffoldsNonThinking)),
                // Google's recommended sampling for Gemma - GGUFs converted before the
                // general.sampling.* convention lack it; container values override these
                new LoadedModel.SamplingDefaults(1.0f, 0.95f, 64, null));
    }
}
