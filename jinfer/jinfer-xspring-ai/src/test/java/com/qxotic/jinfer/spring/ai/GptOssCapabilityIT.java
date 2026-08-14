package com.qxotic.jinfer.spring.ai;

/** {@link AbstractCapabilityIT} against gpt-oss (Harmony channels). */
class GptOssCapabilityIT extends AbstractCapabilityIT {

    @Override
    String modelRef() {
        return "hf.co/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf";
    }
}
