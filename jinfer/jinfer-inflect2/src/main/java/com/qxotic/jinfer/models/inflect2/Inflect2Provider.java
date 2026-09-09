package com.qxotic.jinfer.models.inflect2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * {@link ModelProvider} service: the Inflect2 port's arch-dispatch entry, so {@code
 * Models.loadSpeech} finds it without any consumer naming this port. A speech-only family.
 *
 * <p>Defaults only, which is the whole point of dispatch: the front end comes from the ladder (the
 * attached lexicon, else one beside the GGUF, then the classpath, then espeak-ng) and the family's
 * own knobs - variation, seed, word overrides - are on {@link InflectTTS}, for a caller that has
 * chosen to name it.
 */
public final class Inflect2Provider implements ModelProvider {

    @Override
    public Set<String> architectures() {
        return Set.of("inflect-v2");
    }

    /** The pronunciation lexicon: what turns text into phonemes without an external process. */
    @Override
    public Map<String, String> companionFiles() {
        return Map.of("lexicon", "lexicon");
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        return InflectTTS.load(channel, gguf, path, arena, companions.get("lexicon"));
    }
}
