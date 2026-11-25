package com.qxotic.jinfer.spring.ai;

/** {@link AbstractCapabilityIT} against Inkling 2 Mini (native codec: block header per call). */
class Inkling2CapabilityIT extends AbstractCapabilityIT {

    @Override
    String modelRef() {
        return "hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf";
    }
}
