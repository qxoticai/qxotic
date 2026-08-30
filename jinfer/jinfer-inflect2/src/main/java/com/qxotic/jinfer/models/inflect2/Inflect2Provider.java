package com.qxotic.jinfer.models.inflect2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

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
        return architectures().contains(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("inflect-v2");
    }

    /** The pronunciation lexicon: what turns text into phonemes without an external process. */
    @Override
    public Map<String, String> companionFiles() {
        return Map.of("phonemes", "lexicon");
    }

    /** A speech-only family: nothing to generate with - {@code Models.loadSpeech} is the door. */
    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer) {
        throw new UnsupportedOperationException(
                "'inflect2' is a speech-only family - load it with Models.loadSpeech");
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) throws IOException {
        return InflectTTS.load(fileChannel, gguf, path, arena);
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions)
            throws IOException {
        Path lexicon = companions.get("phonemes");
        return lexicon == null
                ? InflectTTS.load(fileChannel, gguf, path, arena)
                : InflectTTS.load(fileChannel, gguf, path, arena, lexicon);
    }
}
