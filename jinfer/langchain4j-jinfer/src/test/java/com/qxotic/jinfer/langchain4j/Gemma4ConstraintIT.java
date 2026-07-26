package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/**
 * {@link AbstractConstraintIT} against Gemma 4 (no think vocabulary: constrained from token zero).
 */
class Gemma4ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.gemma4Model", ModelFixture.GEMMA4_E2B_QAT_Q4.path().toString()));
    }
}
