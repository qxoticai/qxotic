package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractMediaIT} against Qwen3.5-4B with its Qwen3-VL projector attached. */
class Qwen35MediaIT extends AbstractMediaIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf");
    }

    @Override
    Path mediaCompanion() {
        return TestModels.require("hf.co/unsloth/Qwen3.5-4B-GGUF/mmproj-F32.gguf");
    }
}
