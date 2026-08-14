package com.qxotic.jinfer.spring.ai;

/** {@link AbstractCapabilityIT} against Qwen3.5 (XML call payloads, prompt-opened think). */
class Qwen35CapabilityIT extends AbstractCapabilityIT {

    @Override
    String modelRef() {
        return "hf.co/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
    }
}
