package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.hub.ModelRef.Host;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The ref grammar: everything resolution decides before it touches a host. */
class ModelRefTest {

    // ---- the grammar ----

    @Test
    void aBareRepositoryTakesEveryDefault() {
        ModelRef ref = ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF");
        assertEquals(Host.HF, ref.host());
        assertEquals("unsloth", ref.owner());
        assertEquals("gemma-4-E2B-it-GGUF", ref.repo());
        assertEquals("", ref.location());
        assertNull(ref.quant(), "an unwritten quant stays null: it is what permits the fallback");
        assertEquals("Q4_K_M", ref.quantOrDefault());
        assertEquals("main", ref.revisionOrDefault());
    }

    @Test
    void theTagIsTheQuant() {
        assertEquals("Q8_0", ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0").quant());
    }

    @Test
    void thePathIsTheLocation() {
        ModelRef file = ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        assertEquals("mmproj-F32.gguf", file.location());
        assertNull(file.quant());

        ModelRef folder = ModelRef.parse("hf.co/ggml-org/models/bert-bge-small:F16");
        assertEquals("bert-bge-small", folder.location());
        assertEquals("F16", folder.quant());

        ModelRef deep = ModelRef.parse("hf.co/ggml-org/models/bert-bge-small/ggml-model-f16.gguf");
        assertEquals("bert-bge-small/ggml-model-f16.gguf", deep.location());
    }

    @Test
    void theRevisionAttachesToTheRepository() {
        ModelRef ref = ModelRef.parse("hf.co/ggml-org/models@a1b2c3d/bert-bge-small:F16");
        assertEquals("models", ref.repo());
        assertEquals("a1b2c3d", ref.revision());
        assertEquals("bert-bge-small", ref.location());
        assertEquals("F16", ref.quant());
        assertEquals("models@a1b2c3d", ref.cacheRepo(), "a NAMED revision joins the cache path");
    }

    @Test
    void eachHostKeepsItsOwnDefaultBranch() {
        assertEquals("main", ModelRef.parse("hf.co/a/b").revisionOrDefault());
        assertEquals("master", ModelRef.parse("modelscope.cn/a/b").revisionOrDefault());
        // an unnamed revision never reaches the cache path, so the common case stays clean
        assertEquals("b", ModelRef.parse("modelscope.cn/a/b").cacheRepo());
    }

    // ---- what people actually paste ----

    @Test
    void everySpellingOfARepositoryUrlIsTheSameRef() {
        String canonical = ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF").toString();
        for (String spelling :
                new String[] {
                    "HF.CO/unsloth/gemma-4-E2B-it-GGUF",
                    "hf.co/unsloth/gemma-4-E2B-it-GGUF/",
                    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF",
                    "https://www.huggingface.co/unsloth/gemma-4-E2B-it-GGUF",
                }) {
            assertEquals(canonical, ModelRef.parse(spelling).toString(), spelling);
        }
    }

    @Test
    void theHubsOwnViewUrlsNormalize() {
        ModelRef blob =
                ModelRef.parse(
                        "https://hf.co/ggml-org/models/blob/main/bert-bge-small/ggml-model-f16.gguf");
        assertEquals("main", blob.revision());
        assertEquals("bert-bge-small/ggml-model-f16.gguf", blob.location());

        ModelRef download =
                ModelRef.parse(
                        "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/"
                                + "mmproj-F32.gguf?download=true");
        assertEquals("main", download.revision());
        assertEquals("mmproj-F32.gguf", download.location());

        ModelRef tree = ModelRef.parse("https://hf.co/ggml-org/models/tree/main/bert-bge-small");
        assertEquals("main", tree.revision());
        assertEquals("bert-bge-small", tree.location());
    }

    // ---- remote versus local ----

    @Test
    void onlyAKnownHostMakesARef() {
        assertTrue(ModelRef.isRef("hf.co/a/b"));
        assertTrue(ModelRef.isRef("modelscope.cn/a/b:Q8_0"));
        assertTrue(ModelRef.isRef("https://huggingface.co/a/b"));

        assertFalse(ModelRef.isRef("unsloth/gemma-4-E2B-it-GGUF"), "a bare repo is a path");
        assertFalse(ModelRef.isRef("/models/mine.gguf"));
        assertFalse(ModelRef.isRef("./mine.gguf"));
        assertFalse(ModelRef.isRef("mine.gguf"));
        assertFalse(ModelRef.isRef("C:\\models\\mine.gguf"), "a drive letter is not a host");
        assertFalse(ModelRef.isRef("example.org/models/x.gguf"), "an unknown host is a path");
    }

    @Test
    void isRemoteIsThePublicFormOfTheSameRule() {
        assertTrue(ModelStore.isRemote("hf.co/a/b:Q8_0"));
        assertTrue(ModelStore.isRemote("https://huggingface.co/a/b"));
        assertTrue(
                ModelStore.isRemote("https://example.org/models/x.gguf"), "a plain URL is remote");

        assertFalse(ModelStore.isRemote("/models/mine.gguf"));
        assertFalse(ModelStore.isRemote("./hf.co/a/b"), "the explicit-local escape stays local");
        assertFalse(ModelStore.isRemote("example.org/models/x.gguf"), "no scheme, unknown host");
        assertFalse(ModelStore.isRemote(null));
    }

    @Test
    void aBareRepositoryIsRefusedWithTheHostFormInTheMessage() {
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.resolve("unsloth/gemma-4-E2B-it-GGUF"));
        assertTrue(failure.getMessage().contains("hf.co/"), failure.getMessage());
    }

    @Test
    void malformedRefsTeachTheGrammar() {
        for (String bad : new String[] {"hf.co", "hf.co/", "hf.co/owner", "hf.co/owner/:Q8_0"}) {
            var failure =
                    assertThrows(IllegalArgumentException.class, () -> ModelRef.parse(bad), bad);
            assertTrue(failure.getMessage().contains("hf.co/owner/repo"), bad);
        }
    }

    // ---- the cache jail ----

    @Test
    void noRefCanEscapeTheCacheDirectory() {
        for (String escape :
                new String[] {
                    "hf.co/owner/repo/../../../../etc/evil.gguf",
                    "hf.co/../../etc/repo:Q8_0",
                    "hf.co/owner/..:Q8_0",
                    "hf.co/owner/repo@..:Q8_0",
                    "hf.co/owner/repo:..",
                }) {
            assertThrows(IllegalArgumentException.class, () -> ModelRef.parse(escape), escape);
        }
    }

    @Test
    void aListedFileCannotEscapeEither(@TempDir Path root) {
        System.setProperty("jinfer.models", root.toString());
        ModelRef ref = ModelRef.parse("hf.co/owner/repo");
        // a listing is remote input: a hostile repository must not choose where bytes land
        assertThrows(
                IllegalArgumentException.class,
                () -> ModelStore.pathOf(ref, "../../../etc/x.gguf"));
    }

    @Test
    void aStringThePlatformCannotParseAsAPathIsNotAFile() {
        // A NUL is an illegal path on every platform, as a colon is on Windows. Neither may escape
        // out of resolve: an unparseable path is simply not a file.
        var failure =
                assertThrows(
                        IllegalArgumentException.class, () -> ModelStore.resolve("us\0er/repo"));
        assertTrue(failure.getMessage().contains("no such model file"), failure.getMessage());
    }

    // ---- the cache mapping ----

    @Test
    void theCachePathMirrorsTheRef(@TempDir Path root) {
        System.setProperty("jinfer.models", root.toString());
        assertEquals(
                root.resolve("hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"),
                ModelStore.pathOf(
                        ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0"),
                        "gemma-4-E2B-it-Q8_0.gguf"));
        // subfolders survive, so two files of one name in different folders cannot collide
        assertEquals(
                root.resolve("hf.co/ggml-org/models/bert-bge-small/ggml-model-f16.gguf"),
                ModelStore.pathOf(
                        ModelRef.parse("hf.co/ggml-org/models"),
                        "bert-bge-small/ggml-model-f16.gguf"));
        // a named revision folds into the repository directory
        assertEquals(
                root.resolve("hf.co/ggml-org/models@a1b2c3d/x.gguf"),
                ModelStore.pathOf(ModelRef.parse("hf.co/ggml-org/models@a1b2c3d"), "x.gguf"));
    }

    @Test
    void refsRoundTripThroughToString() {
        for (String ref :
                new String[] {
                    "hf.co/unsloth/gemma-4-E2B-it-GGUF",
                    "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0",
                    "hf.co/ggml-org/models/bert-bge-small:F16",
                    "hf.co/ggml-org/models@a1b2c3d/bert-bge-small/x.gguf",
                    "modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0",
                }) {
            assertEquals(ref, ModelRef.parse(ref).toString());
        }
    }

    // ---- policy that is not grammar ----

    @Test
    void aFileAnotherToolDownloadedIsFound(@TempDir Path hub) throws IOException {
        // the HuggingFace hub layout, exactly as `hf download` or `llama-server -hf` leaves it:
        // blobs by hash, snapshots per commit, and refs/<branch> naming the commit
        Path repo = hub.resolve("models--ggml-org--stories15M_MOE");
        Files.createDirectories(repo.resolve("snapshots/abc123def"));
        Files.createDirectories(repo.resolve("refs"));
        Files.writeString(repo.resolve("refs/main"), "abc123def");
        Files.writeString(repo.resolve("snapshots/abc123def/stories15M_MOE-Q8_0.gguf"), "weights");

        assertEquals(
                repo.resolve("snapshots/abc123def"),
                ModelStore.huggingFaceSnapshot(
                        ModelRef.parse("hf.co/ggml-org/stories15M_MOE:Q8_0"), hub),
                "a branch resolves through refs/ to its commit");
        assertEquals(
                repo.resolve("snapshots/abc123def"),
                ModelStore.huggingFaceSnapshot(
                        ModelRef.parse("hf.co/ggml-org/stories15M_MOE@abc123def"), hub),
                "a pinned commit needs no indirection");
        assertEquals(
                repo.resolve("snapshots/abc123def"),
                ModelStore.huggingFaceSnapshot(
                        ModelRef.parse("hf.co/ggml-org/stories15M_MOE/stories15M_MOE-Q8_0.gguf"),
                        hub),
                "an explicit file resolves to the directory cachedIn searches");
        assertNull(
                ModelStore.huggingFaceSnapshot(ModelRef.parse("hf.co/who/else"), hub),
                "a repository that is not there is not a hit");
        assertNull(
                ModelStore.huggingFaceSnapshot(
                        ModelRef.parse("modelscope.cn/ggml-org/stories15M_MOE"), hub),
                "ModelScope keeps its own layout; not ours to read");
    }

    @Test
    void linkPublishesABlobAsARelativeSymlink(@TempDir Path repo) throws IOException {
        Path blob = repo.resolve("blobs").resolve("a".repeat(64));
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, "weights");
        Path dest = repo.resolve("snapshots").resolve("c".repeat(40)).resolve("x.gguf");

        assertEquals(dest, ModelStore.link(blob, dest));
        assertTrue(Files.isSymbolicLink(dest));
        assertFalse(
                Files.readSymbolicLink(dest).isAbsolute(),
                "relative, so the cache can move as a whole");
        assertEquals("weights", Files.readString(dest));
        assertEquals(dest, ModelStore.link(blob, dest), "publishing twice is a no-op");
    }

    @Test
    void hubCacheGgufsAreListedAsRefs(@TempDir Path hub) throws IOException {
        String commit = "c".repeat(40);
        Path repo = hub.resolve("models--ggml-org--stories15M_MOE");
        Path snapshot = repo.resolve("snapshots").resolve(commit);
        Files.createDirectories(snapshot.resolve("sub"));
        Files.createDirectories(repo.resolve("refs"));
        Files.writeString(repo.resolve("refs/main"), commit);
        Files.writeString(snapshot.resolve("model-Q8_0.gguf"), "weights");
        Files.writeString(snapshot.resolve("sub/mmproj-f16.gguf"), "projector");
        Files.writeString(snapshot.resolve("config.json"), "{}"); // the Python stack's litter
        // a snapshot refs/ no longer names is history, not the cache's current answer
        Files.createDirectories(repo.resolve("snapshots").resolve("d".repeat(40)));
        Files.writeString(
                repo.resolve("snapshots").resolve("d".repeat(40)).resolve("old.gguf"), "old");
        // a repository with no refs cannot say which snapshot is current: skipped, not guessed
        Files.createDirectories(hub.resolve("models--who--else/snapshots"));

        assertEquals(
                List.of(
                        "hf.co/ggml-org/stories15M_MOE/model-Q8_0.gguf",
                        "hf.co/ggml-org/stories15M_MOE/sub/mmproj-f16.gguf"),
                ModelStore.huggingFaceCached(hub).stream().map(ModelStore.Cached::ref).toList());
    }

    @Test
    void evictHubSparesABlobStillLinkedElsewhere(@TempDir Path repo) throws IOException {
        Path blob = repo.resolve("blobs").resolve("a".repeat(64));
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, "weights");
        Path first = repo.resolve("snapshots").resolve("c".repeat(40)).resolve("x.gguf");
        Path second = repo.resolve("snapshots").resolve("d".repeat(40)).resolve("x.gguf");
        ModelStore.link(blob, first);
        ModelStore.link(blob, second);

        assertTrue(ModelStore.evictHub(first));
        assertTrue(Files.exists(blob), "another snapshot still links these bytes");
        assertTrue(ModelStore.evictHub(second));
        assertFalse(Files.exists(blob), "the last link took the blob with it");
    }

    @Test
    void aPlainUrlIsCachedUnderItsHostAndPath(@TempDir Path root) {
        System.setProperty("jinfer.models", root.toString());
        System.setProperty("jinfer.offline", "true"); // never fetch: we assert the PATH it picked
        try {
            var offline =
                    assertThrows(
                            IllegalStateException.class,
                            () -> ModelStore.resolve("https://example.org/models/x.gguf"));
            assertTrue(
                    offline.getMessage()
                            .contains(root.resolve("example.org/models/x.gguf").toString()),
                    offline.getMessage());
        } finally {
            System.clearProperty("jinfer.offline");
        }
    }

    @Test
    void aUrlWithNothingToNameTheFileIsRefused() {
        for (String bad :
                new String[] {
                    "https://example.org/models/", "https://example.org", "ftp://example.org/x.gguf"
                }) {
            assertThrows(IllegalArgumentException.class, () -> ModelStore.resolve(bad), bad);
        }
    }

    @Test
    void quantsMatchAsWholeTokensOnly() {
        assertTrue(ModelStore.matchesQuant("gemma-4-E2B-it-Q4_K_M.gguf", "Q4_K_M"));
        assertTrue(ModelStore.matchesQuant("gemma-4-E2B-it-q4_k_m.gguf", "Q4_K_M")); // fold case
        assertTrue(ModelStore.matchesQuant("stories15M_MOE-F16.gguf", "F16"));
        assertTrue(ModelStore.matchesQuant("gemma-4-E2B_q4_0-it.gguf", "Q4_0")); // '_'-delimited
        // the hazards: a neighbouring quant, and a quant of the same family that is NOT this file
        assertFalse(ModelStore.matchesQuant("gemma-4-E2B-it-Q8_0.gguf", "Q4_K_M"));
        assertFalse(ModelStore.matchesQuant("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "Q4_K_M"));
        assertFalse(ModelStore.matchesQuant("gemma-4-E2B-it-Q4_K_M.gguf", "Q4_K_S"));
        // a PREFIX matches on purpose - it then either identifies one file or the caller is shown
        // every candidate, never a silent pick
        assertTrue(ModelStore.matchesQuant("gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "Q4_K"));
    }

    @Test
    void aDownloadThatCannotFitIsRefusedBeforeItStarts(@TempDir Path dir) {
        var failure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                ModelStore.requireDiskSpace(
                                        dir.resolve("huge.gguf"), Long.MAX_VALUE / 2));
        assertTrue(failure.getMessage().contains("free"), failure.getMessage());
        ModelStore.requireDiskSpace(dir.resolve("small.gguf"), 1024); // one that fits is fine
    }

    @Test
    void resumeAccountingKnowsWhatIsLeftToFetch(@TempDir Path dir) throws IOException {
        long chunk = 32L << 20;
        long size = 3 * chunk;
        Path dest = dir.resolve("model.gguf");
        assertEquals(size, Fetch.remainingBytes(dest, size), "nothing on disk yet");

        // a sequential .part appends, so what is left is what is not there
        Path part = dir.resolve("model.gguf.part");
        Files.write(part, new byte[1024]);
        assertEquals(size - 1024, Fetch.remainingBytes(dest, size));

        // a parallel .part is pre-allocated and sparse: only the chunk map knows the truth
        try (RandomAccessFile allocated = new RandomAccessFile(part.toFile(), "rw")) {
            allocated.setLength(size); // as the parallel path pre-allocates it
        }
        Files.write(dir.resolve("model.gguf.part.map"), new byte[] {1, 0, 1});
        assertEquals(chunk, Fetch.remainingBytes(dest, size), "one full chunk of three missing");
    }

    @Test
    void truncateCannotPreAllocate(@TempDir Path dir) throws IOException {
        // the bug this pins: FileChannel.truncate only ever SHRINKS, so it cannot pre-allocate, and
        // a .part shorter than its file makes every parallel resume discard its chunk map
        Path part = dir.resolve("x.part");
        try (FileChannel channel =
                FileChannel.open(part, StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
            channel.truncate(4096);
        }
        assertEquals(0, Files.size(part), "truncate cannot extend");
        try (RandomAccessFile allocated = new RandomAccessFile(part.toFile(), "rw")) {
            allocated.setLength(4096);
        }
        assertEquals(4096, Files.size(part), "setLength can");
    }
}
