package com.qxotic.jinfer.libjinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.hub.ModelNotFoundException;
import com.qxotic.jinfer.hub.ModelStore;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.concurrent.CancellationException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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

    /** A source that names nothing is ENOTFOUND, by the hub's type, not by its wording. */
    @Test
    void aSourceThatNamesNothingIsNotFound(@TempDir Path root) {
        String absent = root.resolve("absent.gguf").toString();
        var missing =
                assertThrows(
                        ModelNotFoundException.class, () -> ModelStore.of(root).resolve(absent));
        assertEquals(Errors.ENOTFOUND, Errors.code(missing));
    }
}
