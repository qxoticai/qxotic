package com.qxotic.toknroll.libtoknroll;

import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.UnsupportedTokenizerException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AccessDeniedException;
import java.nio.file.FileSystemException;
import java.nio.file.NoSuchFileException;
import org.graalvm.nativeimage.c.type.CCharPointer;
import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.nativeimage.c.type.CTypeConversion.CCharPointerHolder;

/**
 * The calling thread's last error, as the header promises: valid until its next toknroll call,
 * which clears it by succeeding.
 */
final class Errors {
    static final int OK = 0, ERROR = -1, EINVAL = -2, ENOTFOUND = -3, EUNSUPPORTED = -4;

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

    /**
     * "" after a success: made once, on first use, as a holder class initializes. Not in Errors'
     * own initializer, which also runs on a JVM (the unit tests), where a C string cannot be made.
     */
    private static final class None {
        static final CCharPointerHolder EMPTY = CTypeConversion.toCString("");
    }

    /** Every entry point starts here, so that a call that succeeds leaves no error behind. */
    static void clear() {
        LAST.get().code = OK;
    }

    static int code(Throwable t) {
        if (t instanceof Failure f) return f.code;
        if (t instanceof java.io.UncheckedIOException u && u.getCause() != null)
            return code(u.getCause());
        if (t instanceof TokenizerLoadException && t.getCause() != null) return code(t.getCause());
        // well-formed, but asking for what toknroll does not implement: another library may load it
        if (t instanceof UnsupportedTokenizerException) return EUNSUPPORTED;
        // a file that does not parse is bad input, as one of no known kind is: bad JSON, a GGUF cut
        if (t instanceof IllegalArgumentException
                || t instanceof NullPointerException
                || t instanceof com.qxotic.format.json.Json.ParseException
                || t instanceof java.io.EOFException) return EINVAL;
        if (t instanceof java.nio.file.NoSuchFileException
                || t instanceof java.io.FileNotFoundException) return ENOTFOUND;
        return ERROR;
    }

    /**
     * A message's bound in bytes: a parser that echoes its input line cannot make one megabytes
     * long.
     */
    private static final int LIMIT = 1024; // ponytail: a fixed bound

    static int set(Throwable t) {
        String message = t.getMessage();
        // a file the system refused, in the library's words, not java.nio's class names
        if (t.getCause() instanceof FileSystemException f && f.toString().equals(message)) t = f;
        if (t instanceof FileSystemException f) message = plain(f);
        else if (message == null || message.isBlank() || t instanceof NullPointerException) {
            // a message-less failure names its origin, the one thing a C caller cannot recover
            StackTraceElement[] frames = t.getStackTrace();
            message = t.getClass().getSimpleName() + (frames.length > 0 ? " at " + frames[0] : "");
        }
        Throwable cause = t.getCause();
        if (cause != null && cause.getMessage() != null && !message.contains(cause.getMessage()))
            message += ": " + cause.getMessage();
        message = bounded(Log.oneLine(message)); // a caller logs it as it comes: one line, bounded
        int code = code(t);
        Last last = LAST.get();
        if (last.message != null) last.message.close();
        last.message = CTypeConversion.toCString(message);
        last.code = code;
        return code;
    }

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

    static int lastCode() {
        return LAST.get().code;
    }

    static CCharPointer last() {
        Last last = LAST.get();
        return last.code != OK ? last.message.get() : None.EMPTY.get();
    }

    private Errors() {}
}
