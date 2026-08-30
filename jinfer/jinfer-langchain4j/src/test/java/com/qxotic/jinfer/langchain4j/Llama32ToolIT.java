package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractToolIT} against Meta Llama 3.2's bare-JSON function-call wire. */
class Llama32ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return TestModels.require(
                "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
    }
}
