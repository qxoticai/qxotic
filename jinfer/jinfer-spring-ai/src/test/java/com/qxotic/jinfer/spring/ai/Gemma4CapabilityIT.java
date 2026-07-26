package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/** {@link AbstractCapabilityIT} against Gemma 4 (compact key:value calls, no think vocabulary). */
class Gemma4CapabilityIT extends AbstractCapabilityIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.gemma4Model", ModelFixture.GEMMA4_E2B_QAT_Q4.path().toString()));
    }
}
