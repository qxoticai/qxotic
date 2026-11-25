package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against Inkling 2 Mini (reasoning block, then a text block). */
class Inkling2ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf");
    }
}
