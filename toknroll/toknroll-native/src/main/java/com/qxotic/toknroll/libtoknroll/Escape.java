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
 * runs. A handler leaves C's static words for the error (toknroll_isolate.c) and quarantines the
 * call's tokenizer, since the catches it skipped may have left what the call was updating
 * half-written.
 */
final class Escape {
    static final int OUT_OF_MEMORY = 1, STACK_OVERFLOW = 2, OTHER = 3;

    /**
     * The thread's error as C's static words for kind, and its message as it now is; {@code
     * escaped} 1 also quarantines the call's tokenizer. No transition: C neither blocks nor calls
     * back.
     */
    @CFunction(value = "toknroll__escaped", transition = CFunction.Transition.NO_TRANSITION)
    static native CCharPointer words(int kind, int escaped);

    @Uninterruptible(reason = "an entry point's exception handler")
    static int kind(Throwable t) {
        return t instanceof OutOfMemoryError
                ? OUT_OF_MEMORY
                : t instanceof StackOverflowError ? STACK_OVERFLOW : OTHER;
    }

    @Uninterruptible(reason = "an entry point's exception handler")
    private static CCharPointer escaped(Throwable t) {
        return words(kind(t), 1);
    }

    /** The handler of the entry points that return an int: TOKNROLL_ERROR. */
    static final class IntFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static int handle(Throwable t) {
            escaped(t);
            return Errors.ERROR;
        }
    }

    /** The handler of the entry points that return a long: TOKNROLL_ERROR. */
    static final class LongFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static long handle(Throwable t) {
            escaped(t);
            return Errors.ERROR;
        }
    }

    /** The handler of the entry points that return a handle: NULL. */
    static final class HandleFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static ObjectHandle handle(Throwable t) {
            escaped(t);
            return WordFactory.nullPointer();
        }
    }

    /** The handler of the entry points that return a C string: the message, as it now is. */
    static final class StringFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static CCharPointer handle(Throwable t) {
            return escaped(t);
        }
    }

    /** The handler of the entry points that return nothing. */
    static final class VoidFailure implements CEntryPoint.ExceptionHandler {
        @Uninterruptible(reason = "an entry point's exception handler")
        static void handle(Throwable t) {
            escaped(t);
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
