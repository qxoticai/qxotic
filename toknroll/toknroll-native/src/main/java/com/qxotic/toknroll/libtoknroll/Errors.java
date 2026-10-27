package com.qxotic.toknroll.libtoknroll;

import com.qxotic.toknroll.TokenizerLoadException;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;

/**
 * The calling thread's last error, as the header promises: valid until its next toknroll call,
 * which clears it by succeeding.
 */
final class Errors {
    static final int OK = 0, ERROR = -1, EINVAL = -2, ENOTFOUND = -3;

    /** A failure that names its own TOKNROLL_E* code. */
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

    static Failure notFound(String message) {
        return new Failure(ENOTFOUND, message);
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
        if (t instanceof Failure f) return f.code;
        if (t instanceof java.io.UncheckedIOException u && u.getCause() != null)
            return code(u.getCause());
        if (t instanceof TokenizerLoadException && t.getCause() != null) {
            int cause = code(t.getCause());
            return cause == ERROR ? ERROR : cause;
        }
        if (t instanceof IllegalArgumentException || t instanceof NullPointerException)
            return EINVAL;
        if (t instanceof java.nio.file.NoSuchFileException
                || t instanceof java.io.FileNotFoundException) return ENOTFOUND;
        return ERROR;
    }

    static int set(Throwable t) {
        String message = t.getMessage();
        if (message == null || message.isBlank() || t instanceof NullPointerException) {
            // a message-less failure names its origin, the one thing a C caller cannot recover
            StackTraceElement[] frames = t.getStackTrace();
            message = t.getClass().getSimpleName() + (frames.length > 0 ? " at " + frames[0] : "");
        }
        Throwable cause = t.getCause();
        if (cause != null && cause.getMessage() != null && !message.contains(cause.getMessage()))
            message += ": " + cause.getMessage();
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
