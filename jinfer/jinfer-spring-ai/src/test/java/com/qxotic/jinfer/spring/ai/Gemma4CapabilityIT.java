package com.qxotic.jinfer.spring.ai;

/** {@link AbstractCapabilityIT} against Gemma 4 (compact key:value calls, no think vocabulary). */
class Gemma4CapabilityIT extends AbstractCapabilityIT {

    @Override
    String modelRef() {
        return "hf.co/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf";
    }
}
