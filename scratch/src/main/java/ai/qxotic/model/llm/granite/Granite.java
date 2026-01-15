package ai.qxotic.model.llm.granite;

import ai.qxotic.model.llm.llama.FloatSpanFactory;
import ai.qxotic.model.llm.llama.Llama;
import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.KernelOps;

public class Granite extends Llama {
    public Granite(Configuration configuration, KernelOps<FloatSpan, FloatMatrixView> kernelOps, FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
