package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
final class KokoroSpeechIntegrationTest {

    @Test
    void loadsThroughDispatchAndSynthesizesRawText() throws Exception {
        var model = TestModels.require("simonfxr/kokoro.cpp-GGUF/kokoro-82m-q8_0.gguf");
        var voice =
                TestModels.require("simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf");
        try (Arena arena = Arena.ofShared()) {
            @SuppressWarnings("unchecked")
            SpeechSynthesisModel<?, ?, Kokoro.State> speech =
                    (SpeechSynthesisModel<?, ?, Kokoro.State>)
                            Models.loadSpeech(model, arena, Map.of("voice", voice));
            try (Kokoro.State state = speech.newState()) {
                Media.Audio audio = speech.speak(state, "Hi.", SpeechOptions.NONE);
                assertEquals(24_000, audio.sampleRate());
                assertEquals(1, audio.channels());
                assertTrue(audio.pcm().length > 2_400);
                double energy = 0;
                for (float sample : audio.pcm()) energy += sample * sample;
                assertTrue(energy / audio.pcm().length > 1e-8, "waveform must not be silent");
            }
        }
    }

    /** The law: for one utterance, the low-level door and the text door are the same samples. */
    @Test
    void phonemizeThenSynthesizeIsSpeak() throws Exception {
        var model = TestModels.require("simonfxr/kokoro.cpp-GGUF/kokoro-82m-q8_0.gguf");
        var voice =
                TestModels.require("simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf");
        try (Arena arena = Arena.ofShared()) {
            @SuppressWarnings("unchecked")
            SpeechSynthesisModel<?, ?, Kokoro.State> speech =
                    (SpeechSynthesisModel<?, ?, Kokoro.State>)
                            Models.loadSpeech(model, arena, Map.of("voice", voice));
            String text = "Hello, world.";
            int[] phonemes = speech.phonemizer().phonemize(text);
            assertTrue(phonemes.length > 5, "the front end produced ids: " + phonemes.length);
            try (Kokoro.State state = speech.newState()) {
                // bit-stable only once the JIT has settled (measured: from the fourth call); a
                // cold pass rounds a predicted duration differently and shifts every sample after
                for (int warm = 0; warm < 4; warm++)
                    speech.synthesize(state, phonemes, SpeechOptions.NONE);
                Media.Audio low = speech.synthesize(state, phonemes, SpeechOptions.NONE);
                Media.Audio high = speech.speak(state, text, SpeechOptions.NONE);
                assertArrayEquals(low.pcm(), high.pcm());

                int[] offTheTable = {49}; // an empty slot in Kokoro's sparse table
                assertThrows(
                        IllegalArgumentException.class,
                        () -> speech.synthesize(state, offTheTable, SpeechOptions.NONE));
                int[] outOfRange = {178};
                assertThrows(
                        IllegalArgumentException.class,
                        () -> speech.synthesize(state, outOfRange, SpeechOptions.NONE));
            }
        }
    }
}
