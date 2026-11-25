package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against Inkling 2 Mini: the native codec declares tools as a system {@code
 * tool_declare} block of sorted-key JSON, and a call is its own block - {@code
 * <|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}} - closed by {@code
 * <|content_model_end_sampling|>}.
 */
class Inkling2ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf");
    }
}
