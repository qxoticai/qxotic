package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against MiniCPM5 (ChatML with the XML function wire): tojson declarations
 * inside trusted {@code <tools>} ids, calls as {@code <function name=...><param name=...>} spans
 * with CDATA values, tool results folded into user turns. {@code REQUIRED} is marker seeding only.
 */
class MiniCpm5ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return TestModels.require("hf.co/openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q8_0.gguf");
    }
}
