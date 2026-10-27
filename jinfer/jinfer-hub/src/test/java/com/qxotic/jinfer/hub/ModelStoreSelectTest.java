package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.UncheckedIOException;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link ModelStore#select}: the file a ref names, as resolve would choose it, without a download.
 */
class ModelStoreSelectTest {
    @TempDir Path root;

    private static FakeSource repository() {
        return new FakeSource("fake")
                .serving(
                        "",
                        new RemoteFile("thing-Q8_0.gguf", 10, null),
                        new RemoteFile("thing-Q4_K_M.gguf", 10, null),
                        new RemoteFile("tokenizer.json", 2, null))
                .serving("sub", new RemoteFile("sub/other.gguf", 10, null));
    }

    @Test
    void aQuantNamesItsFile() {
        FakeSource fake = repository();
        assertEquals(
                "thing-Q8_0.gguf",
                ModelStore.of(root, fake).select("hf.co/jinfer-tests/thing:Q8_0"));
        assertFalse(fake.fetched());
    }

    @Test
    void aPathNamesItselfOnceItExists() {
        ModelStore store = ModelStore.of(root, repository());
        assertEquals("sub/other.gguf", store.select("hf.co/jinfer-tests/thing/sub/other.gguf"));
        assertEquals("tokenizer.json", store.select("hf.co/jinfer-tests/thing/tokenizer.json"));
    }

    @Test
    void aMissingFileOrQuantIsTheRefsFault() {
        ModelStore store = ModelStore.of(root, repository());
        IllegalArgumentException missing =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> store.select("hf.co/jinfer-tests/thing/none.gguf"));
        assertTrue(missing.getMessage().contains("thing-Q8_0.gguf"), missing.getMessage());
        assertThrows(
                IllegalArgumentException.class,
                () -> store.select("hf.co/jinfer-tests/thing:Q2_K"));
    }

    @Test
    void offlineHasNoListingToSelectFrom() {
        UncheckedIOException e =
                assertThrows(
                        UncheckedIOException.class,
                        () -> ModelStore.of(root).select("hf.co/jinfer-tests/thing:Q8_0"));
        assertTrue(e.getMessage().contains("no source"), e.getMessage());
    }
}
