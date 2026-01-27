package ai.qxotic.jota.tensor;

public interface KernelProgramGenerator {
    KernelProgram generate(ExpressionGraph graph);
}
