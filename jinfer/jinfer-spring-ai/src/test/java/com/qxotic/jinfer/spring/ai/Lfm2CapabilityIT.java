package com.qxotic.jinfer.spring.ai;

/** {@link AbstractCapabilityIT} against LFM2.5 (pythonic call spans, self-opened think). */
class Lfm2CapabilityIT extends AbstractCapabilityIT {

    @Override
    String modelRef() {
        return "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf";
    }
}
