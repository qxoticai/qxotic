package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/** {@link AbstractMediaIT} against Inkling 2 Mini with its Q8_0 projector attached (image only). */
class Inkling2MediaIT extends AbstractMediaIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf");
    }

    @Override
    Path mediaCompanion() {
        return TestModels.require(
                "hf.co/ggml-org/Inkling-2-Mini-GGUF/mmproj-Inkling-2-Mini-Q8_0.gguf");
    }
}
