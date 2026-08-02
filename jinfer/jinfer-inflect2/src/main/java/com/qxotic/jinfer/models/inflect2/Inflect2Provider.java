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
    public SpeechModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel, GGUF gguf, java.nio.file.Path path, Arena arena)
            throws IOException {
        return InflectTTS.load(fileChannel, gguf, path, arena);
    }
}
