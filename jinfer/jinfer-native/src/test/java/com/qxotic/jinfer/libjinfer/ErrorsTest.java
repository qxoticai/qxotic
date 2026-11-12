package com.qxotic.jinfer.libjinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
    void aSecondCallAtOnceIsBusy() {
        Exclusive exclusive = new Exclusive("stream");
        exclusive.enter();
        Errors.Failure busy = assertThrows(Errors.Failure.class, exclusive::enter);
        assertEquals(Errors.EBUSY, busy.code);
        exclusive.leave();
        exclusive.enter(); // free again once the first call left
        exclusive.leave();
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

    /**
     * A ref the hub refuses is coded as libtoknroll codes it: a missing repository is not found.
     */
    @Test
    void aHubRefusalIsNotFoundAndAMalformedRefInvalid() {
        IllegalArgumentException missing =
                new IllegalArgumentException(
                        "acme/nothing does not exist, or is gated or private",
                        new IOException("HTTP 404"));
        assertEquals(Errors.ENOTFOUND, Model.refusal(missing).code);
        assertEquals(
                Errors.ENOTFOUND,
                Model.refusal(new IllegalArgumentException("no such model file: 'x'")).code);
        assertEquals(
                Errors.EINVAL, Model.refusal(new IllegalArgumentException("bad ref: a//b")).code);
    }
}
