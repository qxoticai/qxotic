package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The selection flows - which file a ref picks from a repository listing - driven offline through
 * {@link FakeSource}. The live {@code ModelHubIT} proves these against the real APIs; these prove
 * the logic itself: folder probing, quant matching, the menus, and the format policy.
 */
class ModelStoreSelectionTest {

    @Test
    void aPathThatNamesAFileIsServedVerbatim(@TempDir Path root) throws IOException {
        FakeSource source =
                new FakeSource("fake")
                        .serving("sub", new RemoteFile("sub/mmproj-F32.gguf", 7, null))
                        .bytes("projector");

        Path file = ModelStore.of(root, source).resolve("hf.co/acme/repo/sub/mmproj-F32.gguf");

        assertEquals("projector", Files.readString(file));
        assertEquals(
                List.of("sub"),
                source.requestedDirs(),
                "a file in the parent listing is never re-listed as a folder");
    }

    @Test
    void aPathThatNamesAFolderIsSearchedByQuant(@TempDir Path root) throws IOException {
        FakeSource source =
                new FakeSource("fake")
                        .serving("", new RemoteFile("README.md", 5, null))
                        .serving(
                                "sub",
                                new RemoteFile("sub/m-Q4_0.gguf", 4, null),
                                new RemoteFile("sub/m-Q8_0.gguf", 8, null));

        Path file = ModelStore.of(root, source).resolve("hf.co/acme/repo/sub:Q8_0");

        assertEquals(root.resolve("hf.co/acme/repo/sub/m-Q8_0.gguf"), file);
        assertEquals(
                List.of("", "sub"),
                source.requestedDirs(),
                "the parent is probed first, then the path itself as a folder");
    }

    @Test
    void aPathThatNamesNothingGetsTheParentMenu(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake").serving("", new RemoteFile("a-Q8_0.gguf", 1, null));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo/absent:Q8_0"));
        assertTrue(failure.getMessage().contains("no 'absent' in acme/repo"), failure.getMessage());
        assertTrue(failure.getMessage().contains("a-Q8_0.gguf"), failure.getMessage());
    }

    @Test
    void anEmptyFolderMeansThePathNamesNothing(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake")
                        .serving("", new RemoteFile("a-Q8_0.gguf", 1, null))
                        .serving("empty");

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo/empty:Q8_0"));
        assertTrue(failure.getMessage().contains("no 'empty' in acme/repo"), failure.getMessage());
    }

    @Test
    void aQuantOnAnExplicitFileIsRefused(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake").serving("", new RemoteFile("x-Q8_0.gguf", 1, null));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                ModelStore.of(root, source)
                                        .resolve("hf.co/acme/repo/x-Q8_0.gguf:Q8_0"));
        assertTrue(failure.getMessage().contains("already names a file"), failure.getMessage());
    }

    @Test
    void aSingleModelNeedsNoQuant(@TempDir Path root) throws IOException {
        FakeSource source =
                new FakeSource("fake")
                        .serving(
                                "",
                                new RemoteFile("only-BF16.gguf", 9, null),
                                new RemoteFile("mmproj-only-F16.gguf", 2, null));

        Path file = ModelStore.of(root, source).resolve("hf.co/acme/repo");

        assertEquals(
                root.resolve("hf.co/acme/repo/only-BF16.gguf"),
                file,
                "one model and no quant asked: that is the model, whatever it is called -"
                        + " and the companion was never a candidate");
    }

    @Test
    void aBareRepositoryUsesLlamaCppsDefaultQuant(@TempDir Path root) throws IOException {
        FakeSource source =
                new FakeSource("fake")
                        .serving(
                                "",
                                new RemoteFile("model-Q4_K_M.gguf", 4, null),
                                new RemoteFile("model-Q8_0.gguf", 8, null));

        Path file = ModelStore.of(root, source).resolve("hf.co/acme/repo");

        assertEquals(root.resolve("hf.co/acme/repo/model-Q4_K_M.gguf"), file);
    }

    @Test
    void anAmbiguousQuantShowsTheMenuInsteadOfGuessing(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake")
                        .serving(
                                "",
                                new RemoteFile("b-Q8_0.gguf", 2, null),
                                new RemoteFile("a-Q8_0.gguf", 1, null));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo:Q8_0"));
        assertTrue(failure.getMessage().contains("name the one you want"), failure.getMessage());
        assertTrue(
                failure.getMessage().contains("hf.co/acme/repo/a-Q8_0.gguf"), failure.getMessage());
        assertTrue(
                failure.getMessage().contains("hf.co/acme/repo/b-Q8_0.gguf"), failure.getMessage());
    }

    @Test
    void aMissingQuantListsWhatExists(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake").serving("", new RemoteFile("m-Q4_0.gguf", 1, null));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo:Q8_0"));
        assertTrue(failure.getMessage().contains("no Q8_0 in acme/repo"), failure.getMessage());
        assertTrue(failure.getMessage().contains("m-Q4_0.gguf"), failure.getMessage());
    }

    @Test
    void availableQuantsAreTheTagsPastTheSharedStem() {
        assertEquals(
                "BF16, Q4_K_M, Q8_0, QAD-Q4_0",
                ModelStore.quants(
                        List.of(
                                new RemoteFile("LFM2.5-350M-Q8_0.gguf", 1, null),
                                new RemoteFile("LFM2.5-350M-BF16.gguf", 1, null),
                                new RemoteFile("sub/LFM2.5-350M-QAD-Q4_0.gguf", 1, null),
                                new RemoteFile("LFM2.5-350M-Q4_K_M.gguf", 1, null))));
        assertEquals(
                "Q4_0, Q4_K_M",
                ModelStore.quants(
                        List.of(
                                new RemoteFile("Llama-3.2-1B-Q4_K_M.gguf", 1, null),
                                new RemoteFile("Llama-3.2-1B-Q4_0.gguf", 1, null))));
        // no shared stem, or a file that is only the stem: the names say it best
        assertEquals(
                "a-Q8_0.gguf, b-Q4_0.gguf",
                ModelStore.quants(
                        List.of(
                                new RemoteFile("b-Q4_0.gguf", 1, null),
                                new RemoteFile("a-Q8_0.gguf", 1, null))));
        assertEquals("m.gguf", ModelStore.quants(List.of(new RemoteFile("m.gguf", 1, null))));
    }

    @Test
    void aSafetensorsOnlyRepositoryIsRefusedBeforeAnyBytesMove(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake")
                        .serving("", new RemoteFile("model.safetensors", 20_000_000_000L, null));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo"));
        assertTrue(failure.getMessage().contains("safetensors"), failure.getMessage());
        assertTrue(failure.getMessage().contains("convert_hf_to_gguf"), failure.getMessage());
        // toknroll reads it too: no product named
        assertFalse(failure.getMessage().contains("jinfer"), failure.getMessage());
    }

    @Test
    void aSplitPartIsRefusedWithTheMergeInstructions(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake")
                        .serving("", new RemoteFile("m-Q8_0-00001-of-00002.gguf", 1, null));

        var failure =
                assertThrows(
                        UnsupportedOperationException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo:Q8_0"));
        assertTrue(failure.getMessage().contains("llama-gguf-split"), failure.getMessage());
    }

    @Test
    void aQuantIsNeverThePrefixOfAnother(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake").serving("", new RemoteFile("m-Q4_K_XL.gguf", 1, null));

        var failure =
                assertThrows(
                        ModelNotFoundException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo:Q4_K"));
        assertTrue(failure.getMessage().contains("no Q4_K in acme/repo"), failure.getMessage());
        assertTrue(failure.getMessage().contains("m-Q4_K_XL.gguf"), failure.getMessage());
    }

    @Test
    void aQuantIsItsOwnFileNotOneThatEndsInIt(@TempDir Path root) throws IOException {
        FakeSource source =
                new FakeSource("fake")
                        .serving(
                                "",
                                new RemoteFile("m-QAD-Q4_0.gguf", 1, null),
                                new RemoteFile("m-Q4_0.gguf", 1, null),
                                new RemoteFile("m-UD-Q6_K_XL.gguf", 1, null),
                                new RemoteFile("m-Q6_K.gguf", 1, null));
        ModelStore store = ModelStore.of(root, source);

        assertEquals(
                root.resolve("hf.co/acme/m-GGUF/m-Q4_0.gguf"),
                store.resolve("hf.co/acme/m-GGUF:Q4_0"));
        assertEquals(
                root.resolve("hf.co/acme/m-GGUF/m-Q6_K.gguf"),
                store.resolve("hf.co/acme/m-GGUF:Q6_K"));
        assertEquals(
                root.resolve("hf.co/acme/m-GGUF/m-QAD-Q4_0.gguf"),
                store.resolve("hf.co/acme/m-GGUF:QAD-Q4_0"));
    }

    @Test
    void theCacheNeverAnswersAQuantWithAnotherOne(@TempDir Path root) throws IOException {
        Path qad = root.resolve("hf.co/acme/m-GGUF/m-QAD-Q4_0.gguf");
        Files.createDirectories(qad.getParent());
        Files.writeString(qad, "qad");

        assertTrue(
                ModelStore.of(root).find("hf.co/acme/m-GGUF:Q4_0").isEmpty(),
                "Q4_0 is not downloaded: QAD-Q4_0 is another quant, not a stand-in");
        assertEquals(qad, ModelStore.of(root).find("hf.co/acme/m-GGUF:QAD-Q4_0").orElseThrow());

        FakeSource source =
                new FakeSource("fake")
                        .serving(
                                "",
                                new RemoteFile("m-QAD-Q4_0.gguf", 3, null),
                                new RemoteFile("m-Q4_0.gguf", 7, null));
        Path q40 = ModelStore.of(root, source).resolve("hf.co/acme/m-GGUF:Q4_0");
        assertEquals(root.resolve("hf.co/acme/m-GGUF/m-Q4_0.gguf"), q40);
        assertEquals("weights", Files.readString(q40), "downloaded, not the cached QAD-Q4_0");
    }

    @Test
    void aHostileListingCannotEscapeTheCache(@TempDir Path root) {
        FakeSource source =
                new FakeSource("fake").serving("", new RemoteFile("../evil-Q8_0.gguf", 1, null));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, source).resolve("hf.co/acme/repo:Q8_0"));
        assertTrue(failure.getMessage().contains("escape the cache"), failure.getMessage());
    }
}
