package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against Qwen3.5 (prompt-opened think span). */
class Qwen35ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.qwen35Model", ModelFixture.QWEN35_2B_Q8.path().toString()));
    }
}
