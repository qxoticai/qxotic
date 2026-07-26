package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/** {@link AbstractCapabilityIT} against LFM2.5 (pythonic call spans, self-opened think). */
class Lfm2CapabilityIT extends AbstractCapabilityIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty("jinfer.lfm2Model", ModelFixture.LFM25_8B_Q8.path().toString()));
    }
}
