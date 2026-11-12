package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import java.nio.file.Path;
import java.util.Base64;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/** {@link AbstractMediaIT} against LFM3-VL-2B with its projector attached (images, no audio). */
class Lfm3MediaIT extends AbstractMediaIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/LiquidAI/LFM3-VL-2B-GGUF/LFM3-VL-2B-Q8_0.gguf");
    }

    @Override
    Path mediaCompanion() {
        return TestModels.require("hf.co/LiquidAI/LFM3-VL-2B-GGUF/mmproj-LFM3-VL-2B-F16.gguf");
    }

    /** LFM3-VL's mmproj has a vision tower and no audio one: a clip is refused, by its kind. */
    @Test
    void audioIsRefusedByAVisionOnlyProjector() {
        String wav = Base64.getEncoder().encodeToString(toneWav(440, 0.2, 16000));
        var refused =
                Assertions.assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.chat(
                                        UserMessage.from(
                                                AudioContent.from(wav, "audio/wav"),
                                                TextContent.from("Describe this audio."))));
        // the attached mmproj declares a vision tower only: no advice to attach it again
        Assertions.assertTrue(
                refused.getMessage().contains("this model takes no audio"), refused.getMessage());
        Assertions.assertFalse(refused.getMessage().contains("attach"), refused.getMessage());
    }
}
