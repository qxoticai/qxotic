package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;

/**
 * {@link AbstractCoarseCacheIT} on Qwen3.5 (gated-delta-net hybrid: the S-matrix residue is ~2.1MB
 * per linear layer). The Qwen-only artifact round trip and chunk-shape seam live in the langchain4j
 * twin; the byte-level gate lives in Qwen35CacheRun on the model module.
 */
@Tag("integration")
class Qwen35CoarseCacheIT extends AbstractCoarseCacheIT {

    @Override
    Path modelPath() {
        return ModelFixture.QWEN35_2B_Q8.path();
    }
}
