package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against Gemma 4 (compact {@code call:name{...}} syntax with the trusted
 * quote token, one open model turn per round trip, {@code REQUIRED} forcing via {@code
 * <|tool_call>} seed + {@code call:name} pin).
 */
class Gemma4ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.gemma4Model",
                        TestModels.find(
                                        "hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"))
                                .toString()));
    }
}
