package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;

/**
 * {@link AbstractConstraintIT} against Gemma 4 (no think vocabulary: constrained from token zero).
 */
class Gemma4ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return TestModels.require(
                "hf.co/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
    }
}
