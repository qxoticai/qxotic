// Converts English text → blank-interspersed symbol IDs for the Inflect TTS model.
// Implementations: EspeakPhonemizer (native espeak-ng), LexiconPhonemizer (pure Java).
package com.qxotic.jinfer.models.inflect2.frontend;

import java.io.IOException;

public interface Phonemizer {
    /** Convert English text to blank-interspersed model tokens. */
    int[] phonemize(String text) throws IOException;

    /**
     * Best available implementation, or null if none: the bundled lexicon when there is one, else
     * espeak-ng.
     *
     * <p>The lexicon leads because it is a hash lookup while espeak is a SUBPROCESS PER WORD —
     * measured at 55x realtime against 34x on the same binary and text. Espeak led here originally,
     * which made throughput depend on whether espeak-ng happened to be installed, silently and with
     * no way to tell from the output: a self-contained binary got 38% slower the moment a user had
     * it on PATH.
     *
     * <p>The two are alternatives, not layers. Espeak has a letter-to-sound model and can pronounce
     * anything; the lexicon knows only what it was built with and leaves the rest unspoken (it says
     * so on stderr). Ship a lexicon that covers your text, or drop it and take espeak.
     */
    static Phonemizer tryCreate() {
        Phonemizer lexicon = LexiconPhonemizer.tryCreate();
        return lexicon != null ? lexicon : EspeakPhonemizer.tryCreate();
    }
}
