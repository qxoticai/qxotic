package ai.qxotic.jota.tensor;

public interface KernelInput {
    Tensor get(int index);

    Tensor get(String name);

    int size();

    <T> T param(String name, Class<T> type);
}
