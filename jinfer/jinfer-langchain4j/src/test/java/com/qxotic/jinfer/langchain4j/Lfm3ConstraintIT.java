package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against LFM3 (prompt-opened think, XML call wire). */
class Lfm3ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf");
    }
}
