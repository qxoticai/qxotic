package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

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
}
