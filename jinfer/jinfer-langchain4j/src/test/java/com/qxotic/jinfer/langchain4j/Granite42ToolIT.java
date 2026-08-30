package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractToolIT} against Granite 4.2's function/parameter tool-call dialect. */
class Granite42ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/ibm-granite/granite-4.2-3b-GGUF/granite-4.2-3b-Q8_0.gguf");
    }
}
