package com.qxotic.jinfer.langchain4j;

import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against Qwen 3.5: tool declarations go through the Jinja whole-render
 * fallback (the native codec has no tool encode yet) while replies parse natively via the {@code
 * <tool_call>} span grammar - the one family exercising the fallback-prompt + native-parser
 * combination. {@code REQUIRED} is marker seeding only (no pin hook).
 */
class Qwen35ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.qwen35Model",
                        "/home/mukel/Desktop/playground/models/unsloth/Qwen3.5-4B-Q8_0.gguf"));
    }
}
