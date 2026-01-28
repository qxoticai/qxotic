package ai.qxotic.jota.hip;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.ComputeBackend;
import ai.qxotic.jota.tensor.ExecutionStream;
import ai.qxotic.jota.tensor.ExpressionGraph;
import ai.qxotic.jota.tensor.KernelArgs;
import ai.qxotic.jota.tensor.KernelArgsBuilder;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.KernelHarness;
import ai.qxotic.jota.tensor.KernelInterpreter;
import ai.qxotic.jota.tensor.KernelProgram;
import ai.qxotic.jota.tensor.KernelProgramGenerator;
import ai.qxotic.jota.tensor.LaunchConfig;
import ai.qxotic.jota.tensor.LaunchHints;
import ai.qxotic.jota.tensor.ReductionOp;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

final class HipComputeBackend implements ComputeBackend {

    private static final String ENV_HSACO = "HIP_HSACO_PATH";
    private static final String ENV_KERNEL = "HIP_KERNEL_NAME";

    private final Device device;
    private final KernelProgramGenerator generator = new HipKernelProgramGenerator();
    private final HipKernelBackend backend = new HipKernelBackend();
    private final KernelArgsBuilder argsBuilder = new KernelArgsBuilder();
    private final KernelHarness harness = new KernelHarness(generator, backend, argsBuilder);
    private final HipReductionKernelGenerator reductionGenerator =
            new HipReductionKernelGenerator();

    HipComputeBackend(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryView<?> execute(ExpressionGraph graph, List<Tensor> inputs) {
        if (!HipRuntime.isAvailable()) {
            throw new IllegalStateException("HIP runtime not available");
        }
        String hsacoPath = System.getenv(ENV_HSACO);
        String kernelName = System.getenv(ENV_KERNEL);
        DataType rootType = graph.root().dataType();
        Layout layout = graph.root().layout();
        Shape shape = layout.shape();
        if (!layout.stride().equals(Stride.rowMajor(shape))) {
            throw new UnsupportedOperationException("HIP backend requires row-major layout");
        }
        if (graph.reductionRoot().isPresent()) {
            return executeReduction(graph, inputs);
        }
        return executeElementwise(graph, inputs, hsacoPath, kernelName, rootType, shape);
    }

    private MemoryView<?> executeReduction(ExpressionGraph graph, List<Tensor> inputs) {
        ExpressionGraph.ReductionInfo reduction =
                graph.reductionRoot()
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Expected reduction root for HIP sum execution"));
        if (reduction.op() != ReductionOp.SUM
                && reduction.op() != ReductionOp.MIN
                && reduction.op() != ReductionOp.MAX) {
            throw new UnsupportedOperationException(
                    "HIP backend reduction supports sum/min/max only, got: "
                            + reduction.op().name());
        }
        if (graph.root().dataType() != DataType.FP32 && graph.root().dataType() != DataType.FP64) {
            return executeReductionHost(graph, inputs);
        }

        ExpressionGraph inputGraph =
                new ExpressionGraph(reduction.input(), graph.inputs(), graph.inputTensorMap());
        MemoryView<?> inputView = execute(inputGraph, inputs);
        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> deviceInput =
                inputView.memory().device().equals(Device.HIP)
                        ? castDevice(inputView)
                        : toDevice(hipContext, inputView);

        if (!deviceInput.layout().stride().equals(Stride.rowMajor(deviceInput.shape()))) {
            return executeReductionHost(graph, inputs);
        }
        if (deviceInput.dataType() != graph.root().dataType()) {
            return executeReductionHost(graph, inputs);
        }

        Layout layout = graph.root().layout();
        Shape shape = layout.shape();
        long outSize = shape.size();
        if (outSize > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("HIP backend supports int32 sizes only");
        }
        MemoryView<HipDevicePtr> devOut =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(graph.root().dataType(), outSize),
                        graph.root().dataType(),
                        Layout.rowMajor(shape));

        HipReductionKernelGenerator.KernelProgramSpec spec =
                reductionGenerator.generate(reduction, deviceInput.shape());
        KernelProgram program =
                new KernelProgram(
                        KernelProgram.Kind.SOURCE,
                        KernelProgram.Language.HIP,
                        spec.source(),
                        spec.name(),
                        java.util.Map.of());
        KernelCacheKey key = backend.cacheKey(graph);
        KernelArgs args = new KernelArgs().addBuffer(deviceInput).addBuffer(devOut);
        LaunchConfig config = backend.chooseLaunch(graph, LaunchHints.ELEMENTWISE);
        ExecutionStream stream = new ExecutionStream(graph.root().device(), 0L, true);
        harness.execute(program, key, args, config, stream);

        if (shouldReturnHostOutput()) {
            return toHost(hipContext, devOut);
        }
        return devOut;
    }

    private MemoryView<?> executeReductionHost(ExpressionGraph graph, List<Tensor> inputs) {
        MemoryContext<MemorySegment> hostContext =
                (MemoryContext<MemorySegment>)
                        Environment.current().nativeBackend().memoryContext();
        List<MemoryView<MemorySegment>> hostInputs = new ArrayList<>(inputs.size());
        for (int i = 0; i < graph.inputs().size(); i++) {
            Tensor inputTensor = inputs.get(i);
            MemoryView<?> view =
                    inputTensor.tryGetMaterialized().orElseGet(inputTensor::materialize);
            MemoryView<MemorySegment> hostView = toHost(hostContext, view);
            hostInputs.add(hostView);
        }
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment>[] inputArray =
                hostInputs.toArray(size -> (MemoryView<MemorySegment>[]) new MemoryView[size]);

        Layout layout = graph.root().layout();
        Shape shape = layout.shape();
        MemoryView<MemorySegment> hostOutput =
                MemoryView.of(
                        hostContext
                                .memoryAllocator()
                                .allocateMemory(graph.root().dataType(), shape),
                        graph.root().dataType(),
                        layout);
        KernelInterpreter.execute(graph, inputArray, hostOutput);

        if (shouldReturnHostOutput()) {
            return hostOutput;
        }
        HipMemoryContext hipContext = HipMemoryContext.instance();
        return toDevice(hipContext, hostOutput);
    }

    private MemoryView<?> executeElementwise(
            ExpressionGraph graph,
            List<Tensor> inputs,
            String hsacoPath,
            String kernelName,
            DataType rootType,
            Shape shape) {
        long n = shape.size();
        if (n > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("HIP backend supports int32 sizes only");
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> devOut =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(rootType, n),
                        rootType,
                        Layout.rowMajor(shape));

        KernelProgram program;
        if (hsacoPath != null && !hsacoPath.isBlank()) {
            byte[] hsaco = HipKernelSourceLoader.load(hsacoPath);
            if (kernelName == null || kernelName.isBlank()) {
                throw new UnsupportedOperationException("Missing kernel name for HIP execution");
            }
            program =
                    new KernelProgram(
                            KernelProgram.Kind.BINARY,
                            KernelProgram.Language.HIP,
                            hsaco,
                            kernelName,
                            java.util.Map.of());
        } else {
            program = generator.generate(graph);
        }
        KernelCacheKey key = backend.cacheKey(graph);
        KernelArgs args = argsBuilder.build(graph, inputs, devOut);
        LaunchConfig config = backend.chooseLaunch(graph, LaunchHints.ELEMENTWISE);
        ExecutionStream stream = new ExecutionStream(graph.root().device(), 0L, true);
        harness.execute(program, key, args, config, stream);
        if (shouldReturnHostOutput()) {
            return toHost(hipContext, devOut);
        }
        return devOut;
    }

    private static MemoryView<HipDevicePtr> toDevice(
            HipMemoryContext hipContext, MemoryView<?> view) {
        if (view.memory().device().equals(Device.HIP)) {
            @SuppressWarnings("unchecked")
            MemoryView<HipDevicePtr> hipView = (MemoryView<HipDevicePtr>) view;
            return hipView;
        }
        @SuppressWarnings("unchecked")
        MemoryContext<Object> srcContext =
                (MemoryContext<Object>)
                        Environment.current().backend(view.memory().device()).memoryContext();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryView<HipDevicePtr> dst =
                MemoryView.of(
                        hipContext
                                .memoryAllocator()
                                .allocateMemory(view.dataType(), view.shape().size()),
                        view.dataType(),
                        Layout.rowMajor(view.shape()));
        MemoryContext.copy(srcContext, srcView, hipContext, dst);
        return dst;
    }

    private static MemoryView<HipDevicePtr> castDevice(MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> hipView = (MemoryView<HipDevicePtr>) view;
        return hipView;
    }

    private static MemoryView<MemorySegment> toHost(
            MemoryContext<MemorySegment> hostContext, MemoryView<?> view) {
        if (view.memory().base() instanceof MemorySegment
                && view.memory().device().belongsTo(Device.CPU)) {
            @SuppressWarnings("unchecked")
            MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) view;
            return hostView;
        }
        MemoryView<MemorySegment> dst =
                MemoryView.of(
                        hostContext.memoryAllocator().allocateMemory(view.dataType(), view.shape()),
                        view.dataType(),
                        view.layout());
        @SuppressWarnings("unchecked")
        MemoryContext<Object> srcContext =
                (MemoryContext<Object>)
                        Environment.current().backend(view.memory().device()).memoryContext();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        MemoryContext.copy(srcContext, srcView, hostContext, dst);
        return dst;
    }

    private static boolean shouldReturnHostOutput() {
        Device defaultDevice = Environment.current().defaultDevice();
        return defaultDevice.belongsTo(Device.CPU);
    }

    private static MemoryView<?> toHost(
            HipMemoryContext hipContext, MemoryView<HipDevicePtr> view) {
        @SuppressWarnings("unchecked")
        MemoryContext<Object> hostContext =
                (MemoryContext<Object>) Environment.current().nativeBackend().memoryContext();
        @SuppressWarnings("unchecked")
        MemoryView<Object> hostView =
                (MemoryView<Object>)
                        MemoryView.of(
                                hostContext
                                        .memoryAllocator()
                                        .allocateMemory(view.dataType(), view.shape().size()),
                                view.dataType(),
                                Layout.rowMajor(view.shape()));
        MemoryContext.copy(hipContext, view, hostContext, hostView);
        return hostView;
    }
}
