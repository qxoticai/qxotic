package com.qxotic.jinfer.spring.ai;

/** {@link AbstractCapabilityIT} against LFM3 (XML call payloads, prompt-opened think). */
class Lfm3CapabilityIT extends AbstractCapabilityIT {

    @Override
    String modelRef() {
        return "hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf";
    }
}
