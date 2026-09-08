package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
        var model = TestModels.require("simonfxr/kokoro.cpp-GGUF/kokoro-82m-f16.gguf");
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
}
