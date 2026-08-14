package com.qxotic.jinfer.spring.ai;

import org.junit.jupiter.api.Tag;

/**
 * {@link AbstractCoarseCacheIT} on Qwen3.5 (gated-delta-net hybrid: the S-matrix residue is ~2.1MB
 * per linear layer). The Qwen-only artifact round trip and chunk-shape seam live in the langchain4j
 * twin; the byte-level gate lives in Qwen35CacheRun on the model module. Model-gated via {@code
 * TestModels}.
 */
@Tag("integration")
class Qwen35CoarseCacheIT extends AbstractCoarseCacheIT {

    @Override
    String modelRef() {
        return "hf.co/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
    }
}
