package ai.qxotic.model.llm.qwen2;

import ai.qxotic.model.llm.llama.FloatSpanFactory;
import ai.qxotic.model.llm.llama.Llama;
import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.KernelOps;

public class Qwen2 extends Llama {
    public Qwen2(
            Configuration configuration,
            KernelOps<FloatSpan, FloatMatrixView> kernelOps,
            FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
