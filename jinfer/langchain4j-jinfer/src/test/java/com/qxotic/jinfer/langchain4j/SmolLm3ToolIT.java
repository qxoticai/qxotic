package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against SmolLM3 (ChatML with the metadata header): python-repr tool
 * signatures in the system turn, {@code <tool_call>} JSON spans, tool results as user turns. {@code
 * REQUIRED} is marker seeding only (no pin hook).
 */
class SmolLm3ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.smollm3Model", ModelFixture.SMOLLM3_Q4.path().toString()));
    }
}
