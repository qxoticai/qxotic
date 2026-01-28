package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Environment;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

final class OptimizingCallSiteImpl implements OptimizingCallSite {

    private final Function<Tensor, Tensor> function;
    private final Map<TensorSpec, KernelEntry> cache = new ConcurrentHashMap<>();

    OptimizingCallSiteImpl(Function<Tensor, Tensor> function) {
        this.function = Objects.requireNonNull(function, "function");
    }

    @Override
    public Tensor apply(Tensor input) {
        Objects.requireNonNull(input, "input");
        if (input instanceof TraceTensor) {
            return trace(input);
        }
        TensorSpec inputSpec = TensorSpec.from(input);
        KernelEntry entry = cache.computeIfAbsent(inputSpec, spec -> compileFor(input));
        return execute(entry, input);
    }

    private Tensor trace(Tensor input) {
        InputNode inputNode = new InputNode(0, input.dataType(), input.layout(), input.device());
        TraceTensor tracedInput = new TraceTensor(inputNode);
        TracingTensorOps tracingOps = new TracingTensorOps();
        TraceTensor output =
                TensorOpsContext.with(
                        tracingOps,
                        () -> {
                            Tensor result = function.apply(tracedInput);
                            if (result instanceof TraceTensor traceResult) {
                                return traceResult;
                            }
                            throw new IllegalStateException(
                                    "Tracing function must return a traced tensor, got: " + result);
                        });
        return output;
    }

    private KernelEntry compileFor(Tensor input) {
        Tensor traced = Tracer.trace(input, function);
        ExpressionComputation computation =
                (ExpressionComputation)
                        traced.computation()
                                .orElseThrow(
                                        () ->
                                                new IllegalStateException(
                                                        "Expected traced computation"));
        ExpressionGraph graph = computation.graph();
        TensorSpec outputSpec = new TensorSpec(traced.dataType(), traced.layout(), traced.device());
        LayoutSignature signature =
                new LayoutSignature(List.of(TensorSpec.from(input)), outputSpec);
        return new KernelEntry(signature, graph, outputSpec);
    }

    private Tensor execute(KernelEntry entry, Tensor input) {
        ComputeEngine engine =
                Environment.current().backend(entry.outputSpec().device()).computeEngine();
        ComputeBackend backend = engine.backendFor(entry.outputSpec().device());
        return Tensor.of(backend.execute(entry.graph(), List.of(input)));
    }

    private record KernelEntry(
            LayoutSignature signature, ExpressionGraph graph, TensorSpec outputSpec) {}
}
