package com.qxotic.jinfer.spring.ai;

import org.junit.jupiter.api.Tag;

/**
 * {@link AbstractCoarseCacheIT} on Nemotron-H (Mamba2 hybrid: ~90MB SSM residue per block at 30B
 * dims). Model-gated via {@code TestModels}. (30B - slow to load)
 */
@Tag("integration")
class JinferCoarseCacheIT extends AbstractCoarseCacheIT {

    @Override
    String modelRef() {
        return "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf";
    }
}
