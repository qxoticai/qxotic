package com.qxotic.toknroll.libtoknroll;

import com.qxotic.format.json.Json;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.UnsupportedTokenizerException;
import java.io.EOFException;
import java.io.FileNotFoundException;
import java.io.UncheckedIOException;
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
 * The calling thread's last error, valid until its next toknroll call or thread exit. A successful
 * call clears it.
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

    /** The thread's last result code; message storage belongs to native TLS. */
    private static final class Last {
        int code = OK;
    }

    private static final ThreadLocal<Last> LAST = ThreadLocal.withInitial(Last::new);

    @CFunction("toknroll__error_buffer")
    private static native CCharPointer messageBuffer(int allocate);

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
        if (t instanceof UncheckedIOException u && u.getCause() != null) return code(u.getCause());
        if (t instanceof TokenizerLoadException && t.getCause() != null) return code(t.getCause());
        // well-formed, but asking for what toknroll does not implement: another library may load it
        if (t instanceof UnsupportedTokenizerException) return EUNSUPPORTED;
        // a file that does not parse is bad input, as one of no known kind is: bad JSON, a GGUF cut
        if (t instanceof IllegalArgumentException
                || t instanceof NullPointerException
                || t instanceof Json.ParseException
                || t instanceof EOFException) return EINVAL;
        if (t instanceof NoSuchFileException || t instanceof FileNotFoundException)
            return ENOTFOUND;
        return ERROR;
    }

    /**
     * A message's bound in bytes: a parser that echoes its input line cannot make one megabytes
     * long.
     */
    private static final int LIMIT = 1024; // ponytail: a fixed bound

    /**
     * Out of memory, in words made when Errors initializes, at the first call: a full heap leaves
     * nothing to build them with, and a second OutOfMemoryError out of an entry point aborts the
     * process. The runtime's own words name -Xmx, a flag it does not take here.
     */
    private static final byte[] OUT_OF_MEMORY =
            ("out of memory: the runtime's heap is full; raise it with -XX:MaxHeapSize=<size>"
                            + " through configure or "
                            + Log.optionsVariable()
                            + " (a vocabulary of 262k tokens needs about 256 MB)")
                    .getBytes(StandardCharsets.UTF_8);

    static int set(Throwable t) {
        if (t instanceof OutOfMemoryError) return outOfMemory();
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
        CCharPointer buffer = messageBuffer(1);
        if (buffer.isNonNull())
            CTypeConversion.toCString(
                    message, StandardCharsets.UTF_8, buffer, WordFactory.unsigned(LIMIT + 1));
        // If allocation failed, reads return static fallback text with the original error code.
        last.code = code;
        return code;
    }

    /** An OutOfMemoryError, recorded without allocating: the words copied into the C buffer. */
    private static int outOfMemory() {
        CCharPointer buffer = messageBuffer(1);
        if (buffer.isNonNull()) {
            for (int i = 0; i < OUT_OF_MEMORY.length; i++) buffer.write(i, OUT_OF_MEMORY[i]);
            buffer.write(OUT_OF_MEMORY.length, (byte) 0);
        }
        LAST.get().code = ERROR;
        return ERROR;
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
        return last.code != OK ? messageBuffer(0) : None.EMPTY.get();
    }

    private Errors() {}
}
