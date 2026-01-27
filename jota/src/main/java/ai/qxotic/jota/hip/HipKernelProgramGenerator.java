package ai.qxotic.jota.hip;

import ai.qxotic.jota.tensor.ExpressionGraph;
import ai.qxotic.jota.tensor.HipElementwiseKernelGenerator;
import ai.qxotic.jota.tensor.KernelProgram;
import ai.qxotic.jota.tensor.KernelProgramGenerator;
import ai.qxotic.jota.tensor.KernelCacheKey;
import ai.qxotic.jota.tensor.GraphHasher;

final class HipKernelProgramGenerator implements KernelProgramGenerator {
    @Override
    public KernelProgram generate(ExpressionGraph graph) {
        KernelCacheKey key = GraphHasher.hash(graph);
        String kernelName = "hip_fused_" + key.value().substring(0, 12);
        HipElementwiseKernelGenerator generator = new HipElementwiseKernelGenerator();
        HipElementwiseKernelGenerator.GeneratedKernel generated =
                generator.generate(graph, kernelName);
        return new KernelProgram(
                KernelProgram.Kind.SOURCE,
                KernelProgram.Language.HIP,
                generated.source(),
                generated.name(),
                java.util.Map.of());
    }
}
