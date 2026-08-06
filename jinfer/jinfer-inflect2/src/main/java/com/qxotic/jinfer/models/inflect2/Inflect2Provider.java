package com.qxotic.jinfer.models.inflect2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;

/**
 * {@link ModelProvider} service: the Inflect2 port's arch-dispatch entry, so {@code
 * Models.loadSpeech} finds it without any consumer naming this port.
 *
 * <p>Defaults only, which is the whole point of dispatch: the front end comes from the ladder (a
 * lexicon beside the GGUF, then the classpath, then espeak-ng) and the family's own knobs -
 * variation, seed, word overrides - are on {@link InflectTTS}, for a caller that has chosen to name
 * it.
 */
public final class Inflect2Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return architecture.startsWith("inflect");
    }

    @Override
    public java.util.Set<String> architectures() {
        return java.util.Set.of("inflect2"); // representative: supports() matches inflect*
    }

    /** The pronunciation lexicon: what turns text into phonemes without an external process. */
    @Override
    public java.util.Map<String, String> companionFiles() {
        return java.util.Map.of("phonemes", "lexicon");
    }

    @Override
    public SpeechModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel, GGUF gguf, java.nio.file.Path path, Arena arena)
            throws IOException {
        return InflectTTS.load(fileChannel, gguf, path, arena);
    }

    @Override
    public SpeechModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel,
            GGUF gguf,
            java.nio.file.Path path,
            Arena arena,
            java.util.Map<String, java.nio.file.Path> companions)
            throws IOException {
        java.nio.file.Path lexicon = companions.get("phonemes");
        return lexicon == null
                ? InflectTTS.load(fileChannel, gguf, path, arena)
                : InflectTTS.load(fileChannel, gguf, path, arena, lexicon);
    }
}
