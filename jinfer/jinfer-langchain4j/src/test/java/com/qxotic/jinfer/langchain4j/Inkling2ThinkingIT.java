package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractThinkingIT} against Inkling 2 Mini. Its reasoning block opens with {@code
 * <|content_thinking|>} and closes with {@code <|end_message|>}, the same token that ends every
 * other block - which is why the family declares its close as shared.
 */
class Inkling2ThinkingIT extends AbstractThinkingIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf");
    }
}
