package ai.qxotic.jota.tensor;

public interface ComputeStream {
    void enqueue(Runnable task);

    void waitFor(Event event);

    Event record();

    boolean isComplete();

    interface Event {}
}
