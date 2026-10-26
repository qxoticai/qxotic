package com.qxotic.jinfer.libjinfer;

import com.qxotic.jinfer.chat.ModelProvider;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;

/** The calling thread's last error, as the header promises: valid until its next jinfer call. */
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

    private static final ThreadLocal<CCharPointerHolder> LAST = new ThreadLocal<>();
    private static final ThreadLocal<Integer> LAST_CODE = new ThreadLocal<>();
    private static CCharPointerHolder none;

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
        if (t instanceof ExceptionInInitializerError e && e.getCause() != null)
            t = e.getCause(); // a -Djinfer.* value a flags class refused: its own
        // message is the one to report
        String message = t.getMessage();
        if (message == null || message.isBlank() || t instanceof NullPointerException) {
            // a message-less failure names its origin, the one thing a C caller cannot recover
            StackTraceElement[] frames = t.getStackTrace();
            message = t.getClass().getSimpleName() + (frames.length > 0 ? " at " + frames[0] : "");
        }
        set(message);
        int code = code(t);
        LAST_CODE.set(code);
        Log.trace(t); // the full story reaches the host's log callback, at debug level
        return code;
    }

    static int lastCode() {
        Integer code = LAST_CODE.get();
        return code == null ? OK : code;
    }

    static void set(String message) {
        CCharPointerHolder previous = LAST.get();
        if (previous != null) previous.close();
        LAST.set(CTypeConversion.toCString(message));
    }

    static CCharPointer last() {
        CCharPointerHolder holder = LAST.get();
        if (holder != null) return holder.get();
        if (none == null) none = CTypeConversion.toCString("");
        return none.get();
    }

    private Errors() {}
}
