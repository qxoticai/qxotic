package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;

/** {@link AbstractConstraintIT} against LFM2.5 (self-opened think span, pythonic wire). */
class Lfm2ConstraintIT extends AbstractConstraintIT {

    @Override
    Path modelPath() {
        return Path.of(
                System.getProperty("jinfer.lfm2Model", ModelFixture.LFM25_8B_Q8.path().toString()));
    }
}
