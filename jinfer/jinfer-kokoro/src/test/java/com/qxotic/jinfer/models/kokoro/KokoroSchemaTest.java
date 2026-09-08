package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.TensorEntry;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class KokoroSchemaTest {

    @Test
    void readsASeparateLengthIndexedVoicePack() {
        Kokoro.Voice voice = Kokoro.readVoice(voice().build());
        assertEquals(510, voice.maxPhonemes());
        assertEquals(256, voice.styleDimensions());
        Map.of(
                        "af_heart", "en-us",
                        "bf_emma", "en-gb",
                        "ef_dora", "es",
                        "ff_siwis", "fr-fr",
                        "hf_alpha", "hi",
                        "if_sara", "it",
                        "jf_alpha", "ja",
                        "pf_dora", "pt-br",
                        "zf_xiaobei", "cmn")
                .forEach(
                        (name, language) ->
                                assertEquals(
                                        language, Kokoro.readVoiceLanguage(voice(name).build())));
    }

    @Test
    void rejectsIncompatibleVoicePacks() {
        assertVoiceRejected(
                voice().putString("general.architecture", "kokoro"), "voice architecture");
        assertVoiceRejected(
                voice().putTensor(
                                TensorEntry.create(
                                        "voice.pack", new long[] {128, 1, 510}, GGMLType.F32, 0)),
                "shape");
        assertVoiceRejected(
                voice().putTensor(
                                TensorEntry.create(
                                        "voice.pack", new long[] {256, 1, 510}, GGMLType.F16, 0)),
                "F32");
    }

    private static void assertVoiceRejected(Builder builder, String message) {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class, () -> Kokoro.readVoice(builder.build()));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    private static Builder voice() {
        return voice("af_heart");
    }

    private static Builder voice(String name) {
        return Builder.newBuilder()
                .putString("general.architecture", "kokoro-voice")
                .putString("kokoro_voice.name", name)
                .putTensor(
                        TensorEntry.create(
                                "voice.pack", new long[] {256, 1, 510}, GGMLType.F32, 0));
    }
}
