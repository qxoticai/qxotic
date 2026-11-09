package com.qxotic.jinfer.libjinfer;

import com.qxotic.jinfer.chat.ModelProvider;
import java.nio.charset.StandardCharsets;
import java.nio.file.AccessDeniedException;
import java.nio.file.FileSystemException;
import java.nio.file.NoSuchFileException;
import org.graalvm.nativeimage.c.function.CFunction;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;
import org.graalvm.word.WordFactory;

/**
 * The calling thread's last error, valid until its next jinfer call or thread exit. A successful
 * call clears it.
 */
final class Errors {
    private static final System.Logger LOG = System.getLogger("jinfer.native");

    static final int OK = 0,
            DONE = 1,
            ERROR = -1,
            EINVAL = -2,
            ENOTFOUND = -3,
            EUNSUPPORTED = -4,
            ECANCELLED = -5,
            EBUSY = -6;

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

    /** The thread's last result code; message storage belongs to native TLS. */
    private static final class Last {
        int code = OK;
    }

    private static final ThreadLocal<Last> LAST = ThreadLocal.withInitial(Last::new);

    @CFunction("jinfer__error_buffer")
    private static native CCharPointer messageBuffer(int allocate);

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
        // before EINVAL: an IllegalArgumentException too
        if (t instanceof ModelProvider.IncompatibleModelException
                || t instanceof UnsupportedOperationException) return EUNSUPPORTED;
        // a file that does not parse is bad input, as in libtoknroll: bad JSON, a truncated GGUF
        if (t instanceof IllegalArgumentException
                || t instanceof NullPointerException
                || t instanceof com.qxotic.format.json.Json.ParseException
                || t instanceof java.io.EOFException) return EINVAL;
        if (t instanceof java.nio.file.NoSuchFileException
                || t instanceof java.io.FileNotFoundException) return ENOTFOUND;
        if (t instanceof java.util.concurrent.CancellationException) return ECANCELLED;
        return ERROR;
    }

    /**
     * A message's bound in bytes: a parser that echoes its input line cannot make one megabytes
     * long.
     */
    private static final int LIMIT = 1024; // ponytail: a fixed bound

    /** A file the system refused, in its own words: java.nio leaves the reason out of some. */
    private static String plain(FileSystemException f) {
        if (f.getReason() != null) return f.getMessage(); // "file: reason"
        if (f instanceof AccessDeniedException) return f.getFile() + ": permission denied";
        if (f instanceof NoSuchFileException) return f.getFile() + ": does not exist";
        return f.getFile() + ": cannot be read";
    }

    /** At most LIMIT bytes of UTF-8, cut at a character, "…" (3 bytes) marking the cut. */
    private static String bounded(String message) {
        byte[] utf8 = message.getBytes(StandardCharsets.UTF_8);
        if (utf8.length <= LIMIT) return message;
        int end = LIMIT - 3;
        while ((utf8[end] & 0xC0) == 0x80) end--; // never inside a character
        return new String(utf8, 0, end, StandardCharsets.UTF_8) + "…";
    }

    static int set(Throwable t) {
        if (t instanceof java.io.UncheckedIOException u && u.getCause() != null) t = u.getCause();
        // a -Djinfer.* value a flags class refused: its own message is the one to report
        if (t instanceof ExceptionInInitializerError e && e.getCause() != null) t = e.getCause();
        String message = t.getMessage();
        if (t instanceof FileSystemException f) message = plain(f); // not java.nio's class names
        else if (message == null || message.isBlank() || t instanceof NullPointerException) {
            // a message-less failure names its origin, the one thing a C caller cannot recover
            StackTraceElement[] frames = t.getStackTrace();
            message = t.getClass().getSimpleName() + (frames.length > 0 ? " at " + frames[0] : "");
        }
        // a caller logs it as it comes: one line, bounded, as toknroll's
        message = bounded(message.strip().replaceAll("\\s*\\R\\s*", " | "));
        // the full story reaches the host's log callback at debug level, before the error is kept:
        // the callback may call back in (Python frees a dropped handle there), and a call clears it
        LOG.log(System.Logger.Level.DEBUG, "a call failed", t);
        int code = code(t);
        Last last = LAST.get();
        CCharPointer buffer = messageBuffer(1);
        if (buffer.isNonNull())
            CTypeConversion.toCString(
                    message, StandardCharsets.UTF_8, buffer, WordFactory.unsigned(LIMIT + 1));
        // If allocation failed, reads return static fallback text with the original error code.
        last.code = code;
        return code;
    }

    static int lastCode() {
        return LAST.get().code;
    }

    static CCharPointer last() {
        Last last = LAST.get();
        return last.code != OK ? messageBuffer(0) : empty();
    }

    /** "", what a function that returns a string returns when it has none: never NULL. */
    static CCharPointer empty() {
        return None.EMPTY.get();
    }

    /**
     * Made once, on first use, as a holder class initializes: not in Errors' own initializer, which
     * also runs on a JVM (the unit tests), where a C string cannot be made.
     */
    private static final class None {
        static final CCharPointerHolder EMPTY = CTypeConversion.toCString("");
    }

    private Errors() {}
}
