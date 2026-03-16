package com.qxotic.model.llm.llama;

import com.qxotic.span.FloatMatrixView;
import com.qxotic.span.FloatSpan;
import com.qxotic.span.KernelOps;

public final class DefaultKernelOps {

    private static final KernelOps<FloatSpan, FloatMatrixView> DEFAULT =
            new PanamaKernelOps(Util::directAccess);
    private static final FloatSpanFactory<? extends FloatSpan> DEFAULT_SPAN_FACTORY =
            new SimpleSpanFactory();

    public static KernelOps<FloatSpan, FloatMatrixView> getKernelOps() {
        return DEFAULT;
    }

    public static FloatSpanFactory<? extends FloatSpan> getSpanFactory() {
        return DEFAULT_SPAN_FACTORY;
    }
}
