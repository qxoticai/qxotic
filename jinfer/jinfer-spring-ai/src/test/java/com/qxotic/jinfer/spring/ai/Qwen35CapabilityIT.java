package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/** {@link AbstractCapabilityIT} against Qwen3.5 (XML call payloads, prompt-opened think). */
class Qwen35CapabilityIT extends AbstractCapabilityIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty(
                        "jinfer.qwen35Model", ModelFixture.QWEN35_2B_Q8.path().toString()));
    }
}
