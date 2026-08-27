// Converts English text → blank-interspersed symbol IDs for the Inflect TTS model.
// Implementations: EspeakPhonemizer (native espeak-ng), LexiconPhonemizer (pure Java).
package com.qxotic.jinfer.models.inflect2.frontend;

import java.io.IOException;
import java.nio.file.Path;

/**
 * A front end. The two shipped implementations are ALTERNATIVES, not layers: a lexicon is a hash
 * lookup that knows only what it was built with and leaves the rest unspoken (saying so on stderr);
 * espeak has a letter-to-sound model and pronounces anything, at a subprocess per punctuation-free
 * run. Which one a model gets is decided once, at load, by {@code InflectTTS}.
 *
 * <p>Public so a caller can supply their own; the implementations stay package-private behind these
 * factories.
 */
public interface Phonemizer {

    /** Convert English text to blank-interspersed model tokens. */
    int[] phonemize(String text) throws IOException;

    /** The lexicon at {@code path}. Throws when it cannot be read or parsed. */
    static Phonemizer lexicon(Path path) throws IOException {
        return LexiconPhonemizer.read(path);
    }

    /** The lexicon bundled on the classpath, or null when there is none. */
    static Phonemizer bundledLexicon() throws IOException {
        return LexiconPhonemizer.bundled();
    }

    /** espeak-ng from PATH, or null when it is not installed. */
    static Phonemizer espeak() {
        return EspeakPhonemizer.tryCreate();
    }

    /**
     * This front end with espeak-ng covering what it cannot resolve, or itself when espeak-ng is
     * not installed. The lexicon layers espeak under itself: known words stay a deterministic hash
     * lookup, runs containing an unknown word go to espeak's letter-to-sound model, and any
     * remaining gap is guessed by rule - a best-effort pronunciation, never silence.
     */
    default Phonemizer withEspeakFallback() {
        return this;
    }
}
