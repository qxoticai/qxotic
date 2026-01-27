package ai.qxotic.jota.tensor;

public interface KernelOutput {
    Tensor get(int index);

    Tensor get(String name);

    int size();
}
