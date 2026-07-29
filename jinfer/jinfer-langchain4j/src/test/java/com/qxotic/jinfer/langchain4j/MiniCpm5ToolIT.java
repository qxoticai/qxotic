package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/**
 * {@link AbstractToolIT} against MiniCPM5 (ChatML with the XML function wire): tojson declarations
 * inside trusted {@code <tools>} ids, calls as {@code <function name=...><param name=...>} spans
 * with CDATA values, tool results folded into user turns. {@code REQUIRED} is marker seeding only.
 */
class MiniCpm5ToolIT extends AbstractToolIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.minicpmModel", ModelFixture.MINICPM5_1B_Q8.path().toString()));
    }
}
