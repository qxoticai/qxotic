package ai.qxotic.jota.tensor;

public interface KernelInput {
    Tensor get(int index);

    Tensor get(String name);

    KernelInputEntry entry(int index);

    KernelInputEntry entry(String name);

    int size();

    <T> T param(String name, Class<T> type);

    <T> T scalar(int index, Class<T> type);

    <T> T scalar(String name, Class<T> type);
}
