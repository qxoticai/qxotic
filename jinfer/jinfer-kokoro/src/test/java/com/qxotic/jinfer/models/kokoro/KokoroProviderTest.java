package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

final class KokoroProviderTest {

    @Test
    void declaresSpeechArchitectureAndVoiceCompanion() {
        var provider = new KokoroProvider();
        assertEquals(Set.of("kokoro"), provider.architectures());
        assertEquals(Map.of("voice", "voice"), provider.companionFiles());
        assertTrue(provider.supports("kokoro"));
        assertFalse(provider.supports("kokoro-voice"));
    }
}
