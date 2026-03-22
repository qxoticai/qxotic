package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.ir.TIRToLIRLowerer;
import com.qxotic.jota.ir.lir.LIRGraph;
import com.qxotic.jota.ir.lir.LIRInput;
import com.qxotic.jota.ir.lir.LIRStandardPipeline;
import com.qxotic.jota.ir.lir.LIRTextRenderer;
import com.qxotic.jota.ir.lir.ScalarInput;
import com.qxotic.jota.ir.lir.scratch.ScratchAnalysisPass;
import com.qxotic.jota.ir.lir.scratch.ScratchLayout;
import com.qxotic.jota.ir.lir.scratch.ScratchVerificationPass;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.ir.tir.TensorInput;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.ExecutionStream;
import com.qxotic.jota.runtime.KernelArgs;
import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.runtime.LaunchConfig;
import com.qxotic.jota.runtime.clike.LIRKernelArgsBuilder;
import com.qxotic.jota.runtime.mojo.codegen.lir.MojoLirKernelGenerator;
import com.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/** Dedicated Mojo compute engine that owns kernel registration and execution lifecycle. */
final class MojoComputeEngine<T> implements ComputeEngine {

    private final MojoExecutionEngine<T> executionEngine;
    private final MojoMemoryDomain<T> memoryDomain;
    private final KernelService kernelService;
    private final TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
    private final LIRStandardPipeline pipeline = new LIRStandardPipeline();
    private final ScratchAnalysisPass scratchAnalysis = new ScratchAnalysisPass();
    private final ScratchVerificationPass scratchVerification = new ScratchVerificationPass();
    private final MojoLirKernelGenerator mojoLirKernelGenerator = new MojoLirKernelGenerator();
    private final LIRKernelArgsBuilder argsBuilder = new LIRKernelArgsBuilder();
    private final boolean verifyScratch =
            Boolean.parseBoolean(System.getProperty("jota.verifyScratch", "false"));
    private final Device device;

    MojoComputeEngine(MojoExecutionEngine<T> executionEngine) {
        this.executionEngine = executionEngine;
        this.memoryDomain = executionEngine.memoryDomain();
        this.kernelService = executionEngine.kernelService();
        this.device = memoryDomain.device();
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
        List<MemoryView<?>> runtimeInputViews = materializeRuntimeInputViews(graph, inputs);
        List<Layout> inputLayouts = runtimeInputLayouts(graph, runtimeInputViews);
        LIRGraph lirGraph = pipeline.run(lowerer.lower(graph, inputLayouts));
        ScratchLayout scratchLayout = scratchAnalysis.analyze(lirGraph);
        if (verifyScratch) {
            scratchVerification.verifyOrThrow(lirGraph, scratchLayout);
        }
        List<MemoryView<?>> outputs = allocateOutputs(lirGraph, memoryDomain);
        Memory<T> scratch = null;
        if (scratchLayout.requiresScratch()) {
            scratch =
                    memoryDomain
                            .memoryAllocator()
                            .allocateMemory(scratchLayout.alignedTotalByteSize(), 64L);
        }
        KernelCacheKey key = cacheKey(lirGraph, scratchLayout);
        KernelProgram mojoProgram = mojoLirKernelGenerator.generate(lirGraph, scratchLayout, key);
        registerMojoProgram(key, mojoProgram);
        List<Tensor> resolvedInputs =
                resolveInputs(lirGraph, inputs, runtimeInputViews, memoryDomain);
        long scratchPtr = scratch == null ? 0L : executionEngine.addressOf(scratch.base());
        KernelArgs args =
                argsBuilder.buildGroupedWithWorkspaceScalar(
                        lirGraph, resolvedInputs, outputs, scratchPtr, DataType.I64);
        LaunchConfig config = chooseLirLaunchConfig(lirGraph);
        KernelExecutable executable =
                kernelService
                        .loadRegisteredExecutable(key.value())
                        .orElseGet(() -> executionEngine.getOrCompile(mojoProgram, key));
        executable.launch(config, args, new ExecutionStream(device, null, true));

        if (shouldReturnHostOutput()) {
            return toHost(memoryDomain, castDevice(outputs.getFirst()));
        }
        return outputs.getFirst();
    }

    @Override
    public boolean supportsParallelPrecompile() {
        return true;
    }

    @Override
    public void precompile(TIRGraph graph) {
        try {
            LIRGraph lirGraph = pipeline.run(lowerer.lower(graph, graphInputLayouts(graph)));
            ScratchLayout scratchLayout = scratchAnalysis.analyze(lirGraph);
            if (verifyScratch) {
                scratchVerification.verifyOrThrow(lirGraph, scratchLayout);
            }
            KernelCacheKey key = cacheKey(lirGraph, scratchLayout);
            KernelProgram mojoProgram =
                    mojoLirKernelGenerator.generate(lirGraph, scratchLayout, key);
            registerMojoProgram(key, mojoProgram);
            executionEngine.getOrCompile(mojoProgram, key);
        } catch (RuntimeException ignored) {
            // Some scheduled precompile graphs require runtime-resolved input layouts.
        }
    }

    private void registerMojoProgram(KernelCacheKey key, KernelProgram mojoProgram) {
        kernelService.bindKernelName(key.value(), key);
        kernelService.register(mojoProgram, key);
    }

    private static List<Layout> graphInputLayouts(TIRGraph graph) {
        List<Layout> layouts = new ArrayList<>(graph.inputs().size());
        for (var input : graph.inputs()) {
            if (input instanceof com.qxotic.jota.ir.tir.ScalarInput) {
                layouts.add(null);
                continue;
            }
            if (input instanceof TensorInput tensorInput) {
                layouts.add(tensorInput.layout());
                continue;
            }
            throw new IllegalArgumentException(
                    "Unsupported TIR input node for precompile: "
                            + input.getClass().getSimpleName());
        }
        return layouts;
    }

    private static List<MemoryView<?>> materializeRuntimeInputViews(
            TIRGraph graph, List<Tensor> inputs) {
        if (graph.inputs().size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "Expected " + graph.inputs().size() + " inputs but got " + inputs.size());
        }
        List<MemoryView<?>> materialized = new ArrayList<>(inputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            if (graph.inputs().get(i) instanceof com.qxotic.jota.ir.tir.ScalarInput) {
                materialized.add(null);
                continue;
            }
            materialized.add(inputs.get(i).materialize());
        }
        return materialized;
    }

    private static List<Layout> runtimeInputLayouts(
            TIRGraph graph, List<MemoryView<?>> runtimeViews) {
        List<Layout> layouts = new ArrayList<>(graph.inputs().size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            if (graph.inputs().get(i) instanceof com.qxotic.jota.ir.tir.ScalarInput) {
                layouts.add(null);
                continue;
            }
            layouts.add(runtimeViews.get(i).layout());
        }
        return layouts;
    }

    private static <T> List<Tensor> resolveInputs(
            LIRGraph graph,
            List<Tensor> originalInputs,
            List<MemoryView<?>> runtimeInputViews,
            MojoMemoryDomain<T> memoryDomain) {
        List<Tensor> resolved = new ArrayList<>(originalInputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            LIRInput input = graph.inputs().get(i);
            if (input instanceof ScalarInput) {
                resolved.add(originalInputs.get(i));
                continue;
            }
            resolved.add(Tensor.of(toDevice(memoryDomain, runtimeInputViews.get(i))));
        }
        return resolved;
    }

    private static <T> MemoryView<T> toDevice(
            MojoMemoryDomain<T> memoryDomain, MemoryView<?> view) {
        if (view.memory().device().belongsTo(DeviceType.MOJO)) {
            @SuppressWarnings("unchecked")
            MemoryView<T> mojoView = (MemoryView<T>) view;
            return mojoView;
        }
        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>) Environment.memoryDomainFor(view.memory().device());
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        BufferSpec spec = computeBufferSpec(view.layout(), view.dataType());
        MemoryView<T> dst =
                MemoryView.of(
                        memoryDomain.memoryAllocator().allocateMemory(spec.byteSize()),
                        spec.byteOffset(),
                        view.dataType(),
                        view.layout());
        MemoryDomain.copy(srcDomain, srcView, memoryDomain, dst);
        return dst;
    }

    private static <T> MemoryView<T> castDevice(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<T> mojoView = (MemoryView<T>) view;
        return mojoView;
    }

    private static <T> MemoryView<MemorySegment> toHost(
            MemoryDomain<T> deviceDomain, MemoryView<T> view) {
        MemoryDomain<MemorySegment> hostDomain = Environment.nativeMemoryDomain();
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        hostDomain
                                .memoryAllocator()
                                .allocateMemory(view.dataType(), view.shape().size()),
                        view.dataType(),
                        Layout.rowMajor(view.shape()));
        MemoryDomain.copy(deviceDomain, view, hostDomain, hostView);
        return hostView;
    }

    private static boolean shouldReturnHostOutput() {
        return Environment.defaultRuntime().supportsNativeRuntimeAlias();
    }

    private static <T> List<MemoryView<?>> allocateOutputs(
            LIRGraph graph, MojoMemoryDomain<T> memoryDomain) {
        List<MemoryView<?>> outputs = new ArrayList<>(graph.outputs().size());
        for (var output : graph.outputs()) {
            Memory<T> memory =
                    memoryDomain
                            .memoryAllocator()
                            .allocateMemory(output.dataType(), output.shape());
            outputs.add(MemoryView.of(memory, 0, output.dataType(), output.layout()));
        }
        return outputs;
    }

    private static BufferSpec computeBufferSpec(Layout layout, DataType dataType) {
        long[] shape = layout.shape().toArray();
        long[] strideBytes = layout.stride().scale(dataType.byteSize()).toArray();
        long minOffset = 0;
        long maxOffset = 0;
        for (int i = 0; i < shape.length; i++) {
            long dim = shape[i];
            if (dim <= 1) {
                continue;
            }
            long span = (dim - 1) * strideBytes[i];
            if (strideBytes[i] >= 0) {
                maxOffset += span;
            } else {
                minOffset += span;
            }
        }
        long byteOffset = -minOffset;
        long byteSize = maxOffset - minOffset + dataType.byteSize();
        return new BufferSpec(byteOffset, byteSize);
    }

    private record BufferSpec(long byteOffset, long byteSize) {}

    private static LaunchConfig chooseLirLaunchConfig(LIRGraph graph) {
        long workSize = 1L;
        if (!graph.outputs().isEmpty()) {
            workSize = graph.outputs().getFirst().size();
        }
        int threads = 256;
        long blocksLong = (workSize + threads - 1) / threads;
        if (blocksLong < 1) {
            blocksLong = 1;
        }
        int blocks = blocksLong > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) blocksLong;
        return new LaunchConfig(blocks, 1, 1, threads, 1, 1, 0, false);
    }

    private static KernelCacheKey cacheKey(LIRGraph graph, ScratchLayout scratchLayout) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(new LIRTextRenderer().render(graph).getBytes(StandardCharsets.UTF_8));
            digest.update(
                    Long.toString(scratchLayout.totalByteSize()).getBytes(StandardCharsets.UTF_8));
            scratchLayout.offsets().entrySet().stream()
                    .sorted(Comparator.comparingInt(entry -> entry.getKey().id()))
                    .forEach(
                            entry -> {
                                digest.update(
                                        Integer.toString(entry.getKey().id())
                                                .getBytes(StandardCharsets.UTF_8));
                                digest.update(
                                        Long.toString(entry.getValue())
                                                .getBytes(StandardCharsets.UTF_8));
                            });
            byte[] hashed = digest.digest();
            StringBuilder builder = new StringBuilder(hashed.length * 2 + 12);
            for (byte value : hashed) {
                builder.append(String.format(Locale.ROOT, "%02x", value));
            }
            builder.append("-mojo-lir-v13");
            return KernelCacheKey.of(builder.toString());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is not available", e);
        }
    }
}
