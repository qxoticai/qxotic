package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
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
                        "jinfer.gemma4Model", ModelFixture.GEMMA4_E2B_Q8.path().toString()));
    }
}
