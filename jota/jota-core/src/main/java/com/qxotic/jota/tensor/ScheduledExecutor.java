package com.qxotic.jota.tensor;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.tir.KernelStep;
import com.qxotic.jota.ir.tir.ScheduleInputRef;
import com.qxotic.jota.ir.tir.ScheduledOutputRef;
import com.qxotic.jota.ir.tir.ScheduledProgram;
import com.qxotic.jota.ir.tir.ValueId;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import com.qxotic.jota.runtime.ComputeEngine;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

final class ScheduledExecutor {

    MemoryView<?> execute(
            ScheduledProgram program, ComputeEngine computeEngine, List<Tensor> graphInputs) {
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(graphInputs, "graphInputs");

        if (!program.steps().isEmpty() && computeEngine == null) {
            throw new IllegalStateException(
                    "No compute engine available for scheduled execution with "
                            + program.steps().size()
                            + " kernel steps");
        }

        precompileStepsInParallelIfSupported(program, computeEngine);

        Map<ValueId, MemoryView<?>> produced = new HashMap<>();
        for (int i = 0; i < program.steps().size(); i++) {
            KernelStep step = program.steps().get(i);
            List<Tensor> stepInputs = resolveStepInputs(step.inputs(), graphInputs, produced);
            MemoryView<?> output = computeEngine.execute(step.graph(), stepInputs);
            boolean isIntermediate = i < program.steps().size() - 1;
            if (isIntermediate) {
                output = ensureDeviceView(output, computeEngine.device());
            }
            produced.put(step.output(), output);
        }

        return switch (program.output()) {
            case ScheduledOutputRef.ValueOutput value -> getProduced(produced, value.valueId());
            case ScheduledOutputRef.TensorInputOutput input ->
                    inputAt(graphInputs, input.inputId()).materialize();
            case ScheduledOutputRef.ScalarInputOutput input ->
                    inputAt(graphInputs, input.inputId()).materialize();
        };
    }

    private static void precompileStepsInParallelIfSupported(
            ScheduledProgram program, ComputeEngine computeEngine) {
        if (computeEngine == null || !computeEngine.supportsParallelPrecompile()) {
            return;
        }
        if (program.steps().size() < 2) {
            return;
        }
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> futures = new ArrayList<>(program.steps().size());
            for (KernelStep step : program.steps()) {
                futures.add(executor.submit(() -> computeEngine.precompile(step.graph())));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while precompiling scheduled kernels", e);
        } catch (ExecutionException e) {
            throw new IllegalStateException("Failed to precompile scheduled kernels", e.getCause());
        }
    }

    private static List<Tensor> resolveStepInputs(
            List<ScheduleInputRef> refs,
            List<Tensor> graphInputs,
            Map<ValueId, MemoryView<?>> produced) {
        List<Tensor> resolved = new ArrayList<>(refs.size());
        for (ScheduleInputRef ref : refs) {
            Tensor tensor =
                    switch (ref) {
                        case ScheduleInputRef.TensorInputRef input ->
                                inputAt(graphInputs, input.inputId());
                        case ScheduleInputRef.ScalarInputRef input ->
                                inputAt(graphInputs, input.inputId());
                        case ScheduleInputRef.ProducedValueRef value ->
                                Tensor.of(getProduced(produced, value.valueId()));
                    };
            resolved.add(tensor);
        }
        return resolved;
    }

    private static Tensor inputAt(List<Tensor> graphInputs, int inputId) {
        if (inputId < 0 || inputId >= graphInputs.size()) {
            throw new IllegalStateException(
                    "Scheduled input id "
                            + inputId
                            + " is out of bounds for "
                            + graphInputs.size()
                            + " provided inputs");
        }
        return graphInputs.get(inputId);
    }

    private static MemoryView<?> getProduced(Map<ValueId, MemoryView<?>> produced, ValueId id) {
        MemoryView<?> view = produced.get(id);
        if (view == null) {
            throw new IllegalStateException("Missing produced value for " + id);
        }
        return view;
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<?> ensureDeviceView(MemoryView<?> view, Device targetDevice) {
        if (view.memory().device().equals(targetDevice)) {
            return view;
        }

        MemoryDomain<Object> srcDomain = sourceDomainFor((Memory<Object>) view.memory());
        MemoryDomain<Object> dstDomain = Environment.memoryDomainFor(targetDevice);
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        Memory<Object> dstMemory =
                dstDomain.memoryAllocator().allocateMemory(view.dataType(), view.shape());
        MemoryView<Object> dstView =
                MemoryView.of(dstMemory, view.dataType(), Layout.rowMajor(view.shape()));
        MemoryDomain.copy(srcDomain, srcView, dstDomain, dstView);
        return dstView;
    }

    @SuppressWarnings("unchecked")
    private static <B> MemoryDomain<B> sourceDomainFor(Memory<B> memory) {
        Object base = memory.base();
        if (base instanceof boolean[]) {
            return (MemoryDomain<B>) DomainFactory.ofBooleans();
        }
        if (base instanceof byte[]) {
            return (MemoryDomain<B>) DomainFactory.ofBytes();
        }
        if (base instanceof short[]) {
            return (MemoryDomain<B>) DomainFactory.ofShorts();
        }
        if (base instanceof int[]) {
            return (MemoryDomain<B>) DomainFactory.ofInts();
        }
        if (base instanceof long[]) {
            return (MemoryDomain<B>) DomainFactory.ofLongs();
        }
        if (base instanceof float[]) {
            return (MemoryDomain<B>) DomainFactory.ofFloats();
        }
        if (base instanceof double[]) {
            return (MemoryDomain<B>) DomainFactory.ofDoubles();
        }
        if (base instanceof ByteBuffer byteBuffer) {
            return (MemoryDomain<B>)
                    DomainFactory.ofByteBuffer(
                            MemoryAllocatorFactory.ofByteBuffer(byteBuffer.isDirect()));
        }
        if (base instanceof MemorySegment) {
            return (MemoryDomain<B>) Environment.nativeMemoryDomain();
        }
        throw new IllegalArgumentException(
                "Unsupported memory backing type: " + base.getClass().getName());
    }
}
