package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against gpt-oss (Harmony channels: analysis free, final bound). */
class GptOssConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.gptossModel", ModelFixture.GPTOSS_20B_Q8.path().toString()));
    }
}
