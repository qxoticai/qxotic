package com.qxotic.toknroll.libtoknroll;

import com.oracle.svm.shared.Uninterruptible;
import org.graalvm.nativeimage.ObjectHandle;
import org.graalvm.nativeimage.c.function.CEntryPoint;
import org.graalvm.nativeimage.c.function.CFunction;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.word.WordFactory;

/**
 * The entry points' last line. Native image may raise an OutOfMemoryError or a StackOverflowError
 * where no catch runs; whatever reaches an entry point goes to its {@code exceptionHandler}, one of
 * the classes below, never to the process. Native image runs a handler where no safepoint may come:
 * so everything here is uninterruptible and allocates nothing, and this class and its handlers are
 * initialized at build time (native-image.properties), with nothing left to initialize when one
 * runs. A handler leaves C's static words for the error (toknroll_isolate.c). What the call was
 * updating needs no repair: the encode path takes no locks and publishes only whole values.
 */
final class Escape {
    static final int OUT_OF_MEMORY = 1, STACK_OVERFLOW = 2, OTHER = 3;

    /**
     * Thrown to end a batch whose task escaped, its words already said (Workers): made once, at
     * build time, so a full heap can still throw it; {@link Errors#set} leaves the words be.
     */
    static final IllegalStateException BATCH = new IllegalStateException("a batch task escaped");

    /**
     * The thread's error as C's static words for kind, and its message as it now is. No transition:
     * C neither blocks nor calls back.
     */
    @CFunction(value = "toknroll__escaped", transition = CFunction.Transition.NO_TRANSITION)
    static native CCharPointer words(int kind);

    @Uninterruptible(reason = "an entry point's exception handler")
    static int kind(Throwable t) {
        return t instanceof OutOfMemoryError
                ? OUT_OF_MEMORY
                : t instanceof StackOverflowError ? STACK_OVERFLOW : OTHER;
    }

    /** The handler of the entry points that return an int: TOKNROLL_ERROR. */
    static final class IntFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static int handle(Throwable t) {
            words(kind(t));
            return Errors.ERROR;
        }
    }

    /** The handler of the entry points that return a long: TOKNROLL_ERROR. */
    static final class LongFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static long handle(Throwable t) {
            words(kind(t));
            return Errors.ERROR;
        }
    }

    /** The handler of the entry points that return a handle: NULL. */
    static final class HandleFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static ObjectHandle handle(Throwable t) {
            words(kind(t));
            return WordFactory.nullPointer();
        }
    }

    /** The handler of the entry points that return a C string: the message, as it now is. */
    static final class StringFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static CCharPointer handle(Throwable t) {
            return words(kind(t));
        }
    }

    /** The handler of the entry points that return nothing. */
    static final class VoidFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static void handle(Throwable t) {
            words(kind(t));
        }
    }

    /**
     * The handler of a batch task (Workers): how it ended, which C counts as done and reports as
     * the batch's escape. It leaves no words: the batch's calling thread says them.
     */
    static final class TaskFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static int handle(Throwable t) {
            return kind(t);
        }
    }

    private Escape() {}
}
