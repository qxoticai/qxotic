package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against LFM2.5 (pythonic call syntax, {@code <|tool_call_start|>} spans,
 * {@code REQUIRED} forcing via marker seed + {@code [name} pin).
 */
class Lfm2ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.lfm2Model",
                        TestModels.find("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf"))
                                .toString()));
    }
}
