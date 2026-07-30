// Converts English text → blank-interspersed symbol IDs for the Inflect TTS model.
// Implementations: EspeakPhonemizer (native espeak-ng), LexiconPhonemizer (pure Java).
package com.qxotic.jinfer.models.inflect2.frontend;

import java.io.IOException;

public interface Phonemizer {
    /** Convert English text to blank-interspersed model tokens. */
    int[] phonemize(String text) throws IOException;

    /** Best available implementation, or null if none. */
    static Phonemizer tryCreate() {
        Phonemizer p = EspeakPhonemizer.tryCreate();
        if (p != null) return p;
        return LexiconPhonemizer.tryCreate();
    }
}
