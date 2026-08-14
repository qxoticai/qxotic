package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractThinkingIT} against MiniCPM5 (prompt-opened think span). */
class MiniCpm5ThinkingIT extends AbstractThinkingIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q8_0.gguf");
    }
}
