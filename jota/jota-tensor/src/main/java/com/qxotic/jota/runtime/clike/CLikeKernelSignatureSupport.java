package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.ir.lir.BufferRef;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import com.qxotic.jota.ir.lir.ScalarInput;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class CLikeKernelSignatureSupport {

    @FunctionalInterface
    public interface InputArgumentRenderer {
        String render(LIRInput input, int inputIndex);
    }

    @FunctionalInterface
    public interface GroupedInputArgumentRenderer {
        String render(LIRInput input, int inputIndex, int argumentSlot);
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
        Collections.addAll(args, trailingArgs);
        return String.join(", ", args);
    }

    public static int inputBufferCount(LIRGraph graph) {
        int count = 0;
        for (LIRInput input : graph.inputs()) {
            if (input instanceof BufferRef) {
                count++;
            } else if (!(input instanceof ScalarInput)) {
                throw new UnsupportedOperationException(
                        "Unsupported input type in kernel signature: " + input.getClass());
            }
        }
        return count;
    }

    public static String renderKernelArgumentListGrouped(
            LIRGraph graph,
            InputArgumentRenderer inputArgumentRenderer,
            OutputArgumentRenderer outputArgumentRenderer,
            String[] workspaceArgs,
            String... trailingArgs) {
        return renderKernelArgumentListGrouped(
                graph,
                (input, inputIndex, ignoredSlot) -> inputArgumentRenderer.render(input, inputIndex),
                outputArgumentRenderer,
                workspaceArgs,
                trailingArgs);
    }

    public static String renderKernelArgumentListGrouped(
            LIRGraph graph,
            GroupedInputArgumentRenderer inputArgumentRenderer,
            OutputArgumentRenderer outputArgumentRenderer,
            String[] workspaceArgs,
            String... trailingArgs) {
        int bufferInputs = inputBufferCount(graph);
        int scalarInputs = graph.inputs().size() - bufferInputs;
        int argCount =
                bufferInputs
                        + graph.outputs().size()
                        + workspaceArgs.length
                        + scalarInputs
                        + trailingArgs.length;
        List<String> args = new ArrayList<>(argCount);
        int argumentSlot = 0;
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            if (input instanceof BufferRef) {
                args.add(inputArgumentRenderer.render(input, i, argumentSlot++));
            }
        }
        for (int i = 0; i < graph.outputs().size(); i++) {
            int slot = bufferInputs + i;
            args.add(outputArgumentRenderer.render(graph.outputs().get(i), i, slot));
        }
        argumentSlot = bufferInputs + graph.outputs().size();
        for (String workspaceArg : workspaceArgs) {
            args.add(workspaceArg);
            argumentSlot++;
        }
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            if (input instanceof ScalarInput) {
                args.add(inputArgumentRenderer.render(input, i, argumentSlot++));
            }
        }
        Collections.addAll(args, trailingArgs);
        return String.join(", ", args);
    }
}
