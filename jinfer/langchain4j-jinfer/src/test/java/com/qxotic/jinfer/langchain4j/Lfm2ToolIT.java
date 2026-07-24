package com.qxotic.jinfer.langchain4j;

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
                        "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf"));
    }
}
