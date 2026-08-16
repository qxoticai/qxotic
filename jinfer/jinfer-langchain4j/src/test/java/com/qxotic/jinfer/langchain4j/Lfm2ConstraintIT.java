package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against LFM2.5 (self-opened think span, pythonic wire). */
class Lfm2ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf");
    }
}
