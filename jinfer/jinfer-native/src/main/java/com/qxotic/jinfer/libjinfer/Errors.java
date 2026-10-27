package com.qxotic.jinfer.libjinfer;

import com.qxotic.jinfer.chat.ModelProvider;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;

/**
 * The calling thread's last error, as the header promises: valid until its next jinfer call, which
 * clears it by succeeding.
 */
final class Errors {
    static final int OK = 0,
            DONE = 1,
            ERROR = -1,
            EINVAL = -2,
            ENOTFOUND = -3,
            EUNSUPPORTED = -4,
            ECANCELLED = -5;

    /** A failure that names its own JINFER_E* code. */
    static final class Failure extends RuntimeException {
        final int code;

        Failure(int code, String message) {
            super(message);
            this.code = code;
        }
    }

    static Failure invalid(String message) {
        return new Failure(EINVAL, message);
    }

    static Failure unsupported(String message) {
        return new Failure(EUNSUPPORTED, message);
    }

    /** A thread's last failure; its message stays allocated, unread, once a call succeeds. */
    private static final class Last {
        int code = OK;
        CCharPointerHolder message;
    }

    private static final ThreadLocal<Last> LAST = ThreadLocal.withInitial(Last::new);
    private static CCharPointerHolder none;

    /** Every entry point starts here, so that a call that succeeds leaves no error behind. */
    static void clear() {
        LAST.get().code = OK;
    }

    static int code(Throwable t) {
        if (t instanceof ExceptionInInitializerError e && e.getCause() != null)
            return code(e.getCause());
        if (t instanceof Failure f) return f.code;
        if (t instanceof java.io.UncheckedIOException u && u.getCause() != null)
            return code(u.getCause());
        if (t instanceof IllegalArgumentException || t instanceof NullPointerException)
            return EINVAL;
        if (t instanceof java.nio.file.NoSuchFileException
                || t instanceof java.io.FileNotFoundException) return ENOTFOUND;
        if (t instanceof ModelProvider.IncompatibleModelException
                || t instanceof UnsupportedOperationException) return EUNSUPPORTED;
        if (t instanceof java.util.concurrent.CancellationException) return ECANCELLED;
        return ERROR;
    }

    static int set(Throwable t) {
        if (t instanceof java.io.UncheckedIOException u && u.getCause() != null) t = u.getCause();
        // a -Djinfer.* value a flags class refused: its own message is the one to report
        if (t instanceof ExceptionInInitializerError e && e.getCause() != null) t = e.getCause();
        String message = t.getMessage();
        if (message == null || message.isBlank() || t instanceof NullPointerException) {
            // a message-less failure names its origin, the one thing a C caller cannot recover
            StackTraceElement[] frames = t.getStackTrace();
            message = t.getClass().getSimpleName() + (frames.length > 0 ? " at " + frames[0] : "");
        }
        // the full story reaches the host's log callback at debug level, before the error is kept:
        // the callback may call back in (Python frees a dropped handle there), and a call clears it
        Log.trace(t);
        int code = code(t);
        Last last = LAST.get();
        if (last.message != null) last.message.close();
        last.message = CTypeConversion.toCString(message);
        last.code = code;
        return code;
    }

    static int lastCode() {
        return LAST.get().code;
    }

    static CCharPointer last() {
        Last last = LAST.get();
        if (last.code != OK) return last.message.get();
        if (none == null) none = CTypeConversion.toCString("");
        return none.get();
    }

    private Errors() {}
}
