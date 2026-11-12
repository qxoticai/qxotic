package com.qxotic.toknroll.libtoknroll;

import java.util.function.IntConsumer;
import org.graalvm.nativeimage.CurrentIsolate;
import org.graalvm.nativeimage.IsolateThread;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.ObjectHandles;
import org.graalvm.nativeimage.ProcessProperties;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.function.CFunction;

/**
 * A batch's tasks, run on the calling thread and on the library's workers: C threads, the pool and
 * its bookkeeping in toknroll_isolate.c, which counts every task done whatever happened in it. A
 * task is one call of {@link #task}, whose ordinary failure is kept here and rethrown, and whose
 * error past every catch (an OutOfMemoryError or StackOverflowError native image raised where no
 * catch ran) ends in its exception handler and comes back as the batch's escape: the batch fails,
 * the caller's tokenizer is quarantined, and nothing waits on a task that cannot finish. In a
 * forked child, which has none of the threads, a batch runs on the calling thread.
 */
final class Workers {
    /** The threads a batch is shared between, the caller's included. */
    static final int THREADS = Runtime.getRuntime().availableProcessors();

    private static final ObjectHandles HANDLES = ObjectHandles.getGlobal();

    /** The process the runtime started in. */
    private static final long PROCESS = ProcessProperties.getProcessID();

    /**
     * An escaped task's error, made once, while there is a heap to make it in, thrown each time.
     */
    private static final OutOfMemoryError OUT_OF_MEMORY =
            new OutOfMemoryError("a batch task ran out of memory");

    private static final StackOverflowError STACK_OVERFLOW =
            new StackOverflowError("a batch task overflowed its stack");

    private static final IllegalStateException ESCAPED =
            new IllegalStateException("a batch task failed past every catch");

    @CFunction("toknroll__batch")
    private static native int batch(IsolateThread self, ObjectHandle tasks, int count, int workers);

    /** A batch's tasks, and the first ordinary failure among them. */
    private static final class Tasks {
        final IntConsumer task;
        volatile Throwable failure;

        Tasks(IntConsumer task) {
            this.task = task;
        }
    }

    /** Runs every task, sharing work across cores only when there is more than one task. */
    static void run(int tasks, IntConsumer task) {
        if (tasks <= 1 || ProcessProperties.getProcessID() != PROCESS) {
            for (int i = 0; i < tasks; i++) task.accept(i);
            return;
        }
        Tasks list = new Tasks(task);
        ObjectHandle handle = HANDLES.create(list);
        int escaped;
        try {
            escaped = batch(CurrentIsolate.getCurrentThread(), handle, tasks, THREADS - 1);
        } finally {
            HANDLES.destroy(handle);
        }
        if (escaped != 0) Escape.words(escaped, 1); // the call's tokenizer is quarantined
        Throwable failure = list.failure;
        if (failure instanceof RuntimeException e) throw e;
        if (failure instanceof Error e) throw e;
        if (failure != null) throw new IllegalStateException(failure);
        if (escaped == Escape.OUT_OF_MEMORY) throw OUT_OF_MEMORY;
        if (escaped == Escape.STACK_OVERFLOW) throw STACK_OVERFLOW;
        if (escaped != 0) throw ESCAPED;
    }

    /** One task of a batch, on a worker or the calling thread: 0, or the handler's escape. */
    @CEntryPoint(name = "toknroll__task", exceptionHandler = Escape.TaskFailure.class)
    static int task(IsolateThread thread, ObjectHandle handle, int task) {
        Tasks list = HANDLES.get(handle);
        Faults.at("task"); // the tests' error that no catch sees
        try {
            list.task.accept(task);
        } catch (Throwable t) {
            if (list.failure == null) list.failure = t;
        }
        return 0;
    }

    private Workers() {}
}
