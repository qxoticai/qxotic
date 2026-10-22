package com.qxotic.jinfer.libjinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.NoSuchFileException;
import java.util.concurrent.CancellationException;
import org.junit.jupiter.api.Test;

class ErrorsTest {

    @Test
    void codesFollowTheHeader() {
        assertEquals(Errors.EINVAL, Errors.code(new IllegalArgumentException("x")));
        assertEquals(Errors.EINVAL, Errors.code(Errors.invalid("x")));
        assertEquals(Errors.ENOTFOUND, Errors.code(new NoSuchFileException("/x")));
        assertEquals(Errors.EUNSUPPORTED, Errors.code(new UnsupportedOperationException()));
        assertEquals(Errors.ECANCELLED, Errors.code(new CancellationException()));
        assertEquals(Errors.ERROR, Errors.code(new IllegalStateException("x")));
    }

    @Test
    void anUncheckedIoExceptionKeepsItsCauseCode() {
        // the hub wraps file failures unchecked; a missing file stays ENOTFOUND through the wrapper
        assertEquals(
                Errors.ENOTFOUND,
                Errors.code(new UncheckedIOException("open", new NoSuchFileException("/x"))));
        assertEquals(
                Errors.ERROR, Errors.code(new UncheckedIOException("io", new IOException("disk"))));
    }
}
