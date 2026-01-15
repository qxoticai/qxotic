package ai.qxotic.model.llm.llama;

import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.KernelOps;
import ai.qxotic.span.FloatMatrixView;

public final class DefaultKernelOps {

    private static final KernelOps<FloatSpan, FloatMatrixView> DEFAULT = new PanamaKernelOps(Util::directAccess);
    private static final FloatSpanFactory<? extends FloatSpan> DEFAULT_SPAN_FACTORY = new SimpleSpanFactory();

    public static KernelOps<FloatSpan, FloatMatrixView> getKernelOps() {
        return DEFAULT;
    }

    public static FloatSpanFactory<? extends FloatSpan> getSpanFactory() {
        return DEFAULT_SPAN_FACTORY;
    }
}
