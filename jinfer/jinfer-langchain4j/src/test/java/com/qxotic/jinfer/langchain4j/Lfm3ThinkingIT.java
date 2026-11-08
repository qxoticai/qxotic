package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractThinkingIT} against LFM3 (a {@code <think>} span the prompt opens). */
class Lfm3ThinkingIT extends AbstractThinkingIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf");
    }
}
