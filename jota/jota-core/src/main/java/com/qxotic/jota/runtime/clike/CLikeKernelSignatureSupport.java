package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import java.util.ArrayList;
import java.util.List;

public final class CLikeKernelSignatureSupport {

    @FunctionalInterface
    public interface InputArgumentRenderer {
        String render(LIRInput input, int inputIndex);
    }

    @FunctionalInterface
    public interface OutputArgumentRenderer {
        String render(BufferRef output, int outputIndex, int argumentSlot);
    }

    private CLikeKernelSignatureSupport() {}

    public static String renderKernelArgumentList(
            LIRGraph graph,
            InputArgumentRenderer inputArgumentRenderer,
            OutputArgumentRenderer outputArgumentRenderer,
            String... trailingArgs) {
        int argCount = graph.inputs().size() + graph.outputs().size() + trailingArgs.length;
        List<String> args = new ArrayList<>(argCount);
        for (int i = 0; i < graph.inputs().size(); i++) {
            args.add(inputArgumentRenderer.render(graph.inputs().get(i), i));
        }
        for (int i = 0; i < graph.outputs().size(); i++) {
            int slot = graph.inputs().size() + i;
            args.add(outputArgumentRenderer.render(graph.outputs().get(i), i, slot));
        }
        for (String trailingArg : trailingArgs) {
            args.add(trailingArg);
        }
        return String.join(", ", args);
    }
}
