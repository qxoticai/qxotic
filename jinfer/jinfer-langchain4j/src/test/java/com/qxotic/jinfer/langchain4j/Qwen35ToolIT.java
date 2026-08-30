package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against Qwen 3.5: declarations, calls and results use the native
 * checkpoint-exact tool wire; replies parse through the same {@code <tool_call>} span grammar.
 */
class Qwen35ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.qwen35Model",
                        TestModels.find("hf.co/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q8_0.gguf"))
                                .toString()));
    }
}
