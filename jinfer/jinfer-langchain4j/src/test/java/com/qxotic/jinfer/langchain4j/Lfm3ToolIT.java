package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractToolIT} against LFM3 (Qwen-style XML calls in {@code <tool_call>} spans). */
class Lfm3ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf");
    }
}
