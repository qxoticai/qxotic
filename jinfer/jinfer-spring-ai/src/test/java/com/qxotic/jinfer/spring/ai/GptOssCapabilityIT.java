package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/** {@link AbstractCapabilityIT} against gpt-oss (Harmony channels). */
class GptOssCapabilityIT extends AbstractCapabilityIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.gptossModel", ModelFixture.GPTOSS_20B_Q8.path().toString()));
    }
}
