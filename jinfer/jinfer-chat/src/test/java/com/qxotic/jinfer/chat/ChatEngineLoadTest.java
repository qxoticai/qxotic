package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUFFormatException;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.cache.PromptCache;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Isolated;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

@Isolated("Exercises JVM and native-image arena selection through a system property")
final class ChatEngineLoadTest {

    @Test
    void oneArgumentConstructorPreservesTheLoadFailure(@TempDir Path directory) {
        Path missing = directory.resolve("missing.gguf");
        UncheckedIOException error =
                assertThrows(UncheckedIOException.class, () -> new ChatEngine(missing));
        assertTrue(error.getMessage().contains(missing.toString()));
        assertInstanceOf(NoSuchFileException.class, error.getCause());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void loadFailuresKeepTheirCause(boolean automaticArena, @TempDir Path directory)
            throws Exception {
        String property = "org.graalvm.nativeimage.imagecode";
        String previous = System.getProperty(property);
        try {
            if (automaticArena) System.setProperty(property, "runtime");
            else System.clearProperty(property);
            assertEquals(!automaticArena, Arenas.sharedArenas());

            Path missing = directory.resolve("missing.gguf");
            UncheckedIOException io =
                    assertThrows(
                            UncheckedIOException.class,
                            () -> new ChatEngine(missing, null, PromptCache.Options.DEFAULTS));
            assertTrue(io.getMessage().contains(missing.toString()));
            assertInstanceOf(NoSuchFileException.class, io.getCause());

            Path invalid = Files.writeString(directory.resolve("invalid.gguf"), "not a GGUF");
            IllegalArgumentException format =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> new ChatEngine(invalid, Map.of(), PromptCache.Options.DEFAULTS));
            assertTrue(format.getMessage().contains(invalid.toString()));
            assertInstanceOf(GGUFFormatException.class, format.getCause());
        } finally {
            if (previous == null) System.clearProperty(property);
            else System.setProperty(property, previous);
        }
    }

    @Test
    void nullCacheOptionsAreRejectedBeforeOpeningTheModel(@TempDir Path directory) {
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new ChatEngine(directory.resolve("missing.gguf"), Map.of(), null));
        assertEquals("null cache options", error.getMessage());
    }
}
