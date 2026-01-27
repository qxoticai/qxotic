package ai.qxotic.jota.tensor;

import ai.qxotic.jota.memory.MemoryView;
import java.util.List;

public final class KernelHarness {
    private final KernelProgramGenerator generator;
    private final KernelBackend backend;
    private final KernelArgsBuilder argsBuilder;

    public KernelHarness(
            KernelProgramGenerator generator, KernelBackend backend, KernelArgsBuilder argsBuilder) {
        this.generator = generator;
        this.backend = backend;
        this.argsBuilder = argsBuilder;
    }

    public MemoryView<?> execute(ExpressionGraph graph, List<Tensor> inputs, MemoryView<?> output) {
        KernelProgram program = generator.generate(graph);
        KernelCacheKey key = backend.cacheKey(graph);
        KernelArgs args = argsBuilder.build(graph, inputs, output);
        LaunchConfig config = backend.chooseLaunch(graph, LaunchHints.ELEMENTWISE);
        ExecutionStream stream = new ExecutionStream(graph.root().device(), 0L, true);
        execute(program, key, args, config, stream);
        return output;
    }

    public void execute(
            KernelProgram program,
            KernelCacheKey key,
            KernelArgs args,
            LaunchConfig config,
            ExecutionStream stream) {
        KernelExecutable exec = backend.getOrCompile(program, key);
        exec.launch(config, args, stream);
    }
}
