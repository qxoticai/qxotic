package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        assertEquals("hf.co", ref.host());
        assertEquals("unsloth", ref.owner());
        assertEquals("gemma-4-E2B-it-GGUF", ref.repo());
        assertEquals("", ref.path());
        assertNull(ref.quant(), "an unwritten quant stays null: it is what permits the fallback");
        assertEquals("main", ref.revisionOrDefault());
    }

    @Test
    void theTagIsTheQuant() {
        assertEquals("Q8_0", ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0").quant());
    }

    @Test
    void aQuantBeforeThePathIsMisplacedNotAComponent() {
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelRef.parse("ibm-granite/granite-4.2-8b-GGUF:Q6_K/x.gguf"));
        assertEquals(
                "repository 'granite-4.2-8b-GGUF:Q6_K' holds a ':': a :quant comes once, last:"
                        + " owner/repo[/path]:quant",
                failure.getMessage());
    }

    @Test
    void thePathIsTheLocation() {
        ModelRef file = ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        assertEquals("mmproj-F32.gguf", file.path());
        assertNull(file.quant());

        ModelRef folder = ModelRef.parse("hf.co/ggml-org/models/bert-bge-small:F16");
        assertEquals("bert-bge-small", folder.path());
        assertEquals("F16", folder.quant());

        ModelRef deep = ModelRef.parse("hf.co/ggml-org/models/bert-bge-small/ggml-model-f16.gguf");
        assertEquals("bert-bge-small/ggml-model-f16.gguf", deep.path());
    }

    @Test
    void theRevisionAttachesToTheRepository() {
        ModelRef ref = ModelRef.parse("hf.co/ggml-org/models@a1b2c3d/bert-bge-small:F16");
        assertEquals("models", ref.repo());
        assertEquals("a1b2c3d", ref.revision());
        assertEquals("bert-bge-small", ref.path());
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

    // ---- what people actually paste - now strict - canonical only ----

    @Test
    void canonicalSpellingsParseAndRoundTrip() {
        String canonical = ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF").toString();
        assertEquals(canonical, ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF").toString());
        assertEquals(canonical, ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF/").toString());
        assertEquals(
                "modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0",
                ModelRef.parse("modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0").toString());
    }

    @Test
    void aSchemeIsNotARef() {
        assertFalse(ModelRef.isRef("https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF"));
        assertFalse(ModelRef.isRef("https://hf.co/unsloth/gemma-4-E2B-it-GGUF"));
    }

    // ---- remote versus local ----

    @Test
    void aKnownHostOrABareRepositoryMakesARef() {
        assertTrue(ModelRef.isRef("hf.co/a/b"));
        assertTrue(ModelRef.isRef("modelscope.cn/a/b:Q8_0"));
        assertTrue(ModelRef.isRef("unsloth/gemma-4-E2B-it-GGUF"), "a bare repo is a ref");

        assertFalse(ModelRef.isRef("https://huggingface.co/a/b"));
        assertFalse(ModelRef.isRef("/models/mine.gguf"));
        assertFalse(ModelRef.isRef("./mine.gguf"));
        assertFalse(ModelRef.isRef("mine.gguf"), "one segment is not owner/repo");
        assertFalse(ModelRef.isRef("C:\\models\\mine.gguf"), "a drive letter is not a host");
        assertFalse(
                ModelRef.isRef("example.org/models/x.gguf"),
                "a dotted first segment is a misspelled host, not an owner");
    }

    @Test
    void aBareRepositoryIsCarriedByHuggingFace() {
        assertEquals(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0",
                ModelRef.parse("unsloth/gemma-4-E2B-it-GGUF:Q8_0").toString());
        assertEquals("hf.co", ModelRef.parse("unsloth/gemma-4-E2B-it-GGUF").host());
        assertEquals("unsloth", ModelRef.parse("unsloth/gemma-4-E2B-it-GGUF").owner());
    }

    @Test
    void aBareRepositoryKeepsTheRestOfTheGrammar() {
        ModelRef ref = ModelRef.parse("LiquidAI/LFM2.5-VL-3B-GGUF@main/vision/gguf:Q8_0");
        assertEquals("hf.co", ref.host());
        assertEquals("LiquidAI", ref.owner());
        assertEquals("LFM2.5-VL-3B-GGUF", ref.repo());
        assertEquals("main", ref.revision());
        assertEquals("vision/gguf", ref.path());
        assertEquals("Q8_0", ref.quant());
    }

    @Test
    void anExplicitHostIsConsumedAsTheHostNotTheOwner() {
        ModelRef ref = ModelRef.parse("modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0");
        assertEquals("modelscope.cn", ref.host(), "the default must not swallow a named host");
        assertEquals("Qwen", ref.owner());
        assertEquals("Qwen3-0.6B-GGUF", ref.repo());
    }

    @Test
    void requireRefRejectsANonRef() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ModelStore.requireRef("https://huggingface.co/a/b"));
        assertThrows(IllegalArgumentException.class, () -> ModelStore.requireRef("/models/x.gguf"));
    }

    @Test
    void aRelativeModelFileIsAPathNotARepository() {
        // the mistake this catches: a relative path handed to model(String) instead of modelPath
        assertFalse(ModelRef.isRef("models/llama.gguf"), "no repository is called llama.gguf");
        assertFalse(ModelRef.isRef("mydir/tokenizer.json"), "nor tokenizer.json");
        assertTrue(ModelRef.isRef("owner/repo/tokenizer.json"), "a file in a repository");
        assertFalse(ModelRef.isRef("out/model.safetensors"));

        // three segments is the file-in-repository form, which must keep working
        assertTrue(ModelRef.isRef("unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf"));
        assertTrue(ModelRef.isRef("hf.co/owner/repo/mmproj.gguf"));
        assertTrue(ModelRef.isRef("LiquidAI/LFM2.5-350M-GGUF:Q8_0"));
    }

    @Test
    void aFileHereWinsOverABareRepository() {
        assertTrue(
                Files.exists(Path.of("src/main")),
                "precondition: tests run with the module root as the working directory");

        assertFalse(ModelRef.isRef("src/main"), "a directory here is not a repository");
        assertFalse(ModelRef.isRef("src/test"), "a directory here is not a repository");
        assertTrue(ModelRef.isRef("src/nothing-by-this-name"), "nothing here: a repository");
    }

    @Test
    void malformedRefsTeachTheGrammar() {
        for (String bad : new String[] {"hf.co", "hf.co/", "hf.co/owner", "hf.co/owner/:Q8_0"}) {
            var failure =
                    assertThrows(IllegalArgumentException.class, () -> ModelRef.parse(bad), bad);
            assertTrue(failure.getMessage().contains("hf.co/owner/repo"), bad);
        }
        var emptyRevision =
                assertThrows(IllegalArgumentException.class, () -> ModelRef.parse("owner/repo:@"));
        assertEquals(
                "empty revision after '@' in 'owner/repo:@': name a branch, tag or commit",
                emptyRevision.getMessage());
        var emptyQuant =
                assertThrows(IllegalArgumentException.class, () -> ModelRef.parse("owner/repo:"));
        assertEquals(
                "empty quant after ':' in 'owner/repo:': name one, as owner/repo:Q4_K_M",
                emptyQuant.getMessage());
        var refsRevision =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelRef.parse("owner/repo@refs/pr/3/tokenizer.json"));
        assertTrue(
                refsRevision.getMessage().startsWith("a revision is a branch"),
                refsRevision.getMessage());
    }

    @Test
    void aFileNamedTwiceIsRefusedBeforeAnyFetch() {
        var twice =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelRef.parse("owner/repo/sub/model-Q8_0.gguf:Q4_0"));
        assertTrue(twice.getMessage().contains("already names a file"), twice.getMessage());
        assertEquals("Q8_0", ModelRef.parse("owner/repo/sub:Q8_0").quant(), "a folder and a quant");
    }

    @Test
    void aNameAUrlWouldMisreadIsRefused() {
        // a hub name is spliced into a URL as it is: nothing it would read as syntax, a space
        // included
        for (String bad :
                new String[] {
                    "a b/c",
                    "owner/re po",
                    "owner/repo@ma in",
                    "owner/repo:Q8 0",
                    "owner/repo#frag",
                    "owner/repo?x=1",
                    "owner/re%2Fpo",
                    "owner/repo\tx",
                    "owner/<repo>",
                    "owner/repo@main?x",
                    "owner/repo/a|b.gguf",
                    "owner/repo/a\u0001b.gguf",
                }) {
            var failure =
                    assertThrows(IllegalArgumentException.class, () -> ModelRef.parse(bad), bad);
            assertTrue(
                    failure.getMessage().contains("no hub name has"),
                    bad + ": " + failure.getMessage());
        }
        assertEquals("Qwen2.5-0.5B_x+y", ModelRef.parse("Qwen/Qwen2.5-0.5B_x+y").repo());
        // a pull request's revision is spelled with %; a file's path may hold a space, a # or a %
        assertEquals("refs%2Fpr%2F1", ModelRef.parse("owner/repo@refs%2Fpr%2F1").revision());
        assertEquals(
                "dir/my model #2 100%.gguf",
                ModelRef.parse("owner/repo/dir/my model #2 100%.gguf").path());
        assertEquals("日本/モデル.gguf", ModelRef.parse("owner/repo/日本/モデル.gguf").path());
    }

    @Test
    void aFilePathIsPercentEncodedIntoTheUrl() {
        assertEquals(
                "dir/my%20model%20%232%20100%25.gguf",
                RepositorySource.encoded("dir/my model #2 100%.gguf"));
        assertEquals("a-b_c.d~e/F9.gguf", RepositorySource.encoded("a-b_c.d~e/F9.gguf"));
        assertEquals("%E6%97%A5.gguf", RepositorySource.encoded("日.gguf"));
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
                () -> ModelStore.standard().pathOf(ref, "../../../etc/x.gguf"));
    }

    @Test
    void aStringThePlatformCannotParseAsAPathIsRefused() {
        // A NUL is an illegal path on every platform, as a colon is on Windows. Neither may escape
        // out of resolve: the segment check refuses it before anything reaches the network.
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.standard().resolve("us\0er/repo"));
        assertTrue(failure.getMessage().contains("single path component"), failure.getMessage());
    }

    // ---- the cache mapping ----

    @Test
    void theCachePathMirrorsTheRef(@TempDir Path root) {
        System.setProperty("jinfer.models", root.toString());
        assertEquals(
                root.resolve("hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"),
                ModelStore.standard()
                        .pathOf(
                                ModelRef.parse("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0"),
                                "gemma-4-E2B-it-Q8_0.gguf"));
        // subfolders survive, so two files of one name in different folders cannot collide
        assertEquals(
                root.resolve("hf.co/ggml-org/models/bert-bge-small/ggml-model-f16.gguf"),
                ModelStore.standard()
                        .pathOf(
                                ModelRef.parse("hf.co/ggml-org/models"),
                                "bert-bge-small/ggml-model-f16.gguf"));
        // a named revision folds into the repository directory
        assertEquals(
                root.resolve("hf.co/ggml-org/models@a1b2c3d/x.gguf"),
                ModelStore.standard()
                        .pathOf(ModelRef.parse("hf.co/ggml-org/models@a1b2c3d"), "x.gguf"));
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
        // the Hugging Face hub layout, exactly as `hf download` or `llama-server -hf` leaves it:
        // blobs by hash, snapshots per commit, and refs/<branch> naming the commit
        Path repo = hub.resolve("models--ggml-org--stories15M_MOE");
        Files.createDirectories(repo.resolve("snapshots/abc123def"));
        Files.createDirectories(repo.resolve("refs"));
        Files.writeString(repo.resolve("refs/main"), "abc123def");
        Files.writeString(repo.resolve("snapshots/abc123def/stories15M_MOE-Q8_0.gguf"), "weights");

        assertEquals(
                repo.resolve("snapshots/abc123def"),
                Hub.cachedPath(ModelRef.parse("hf.co/ggml-org/stories15M_MOE:Q8_0"), hub),
                "a branch resolves through refs/ to its commit");
        assertEquals(
                repo.resolve("snapshots/abc123def"),
                Hub.cachedPath(ModelRef.parse("hf.co/GGML-org/stories15m_moe:Q8_0"), hub),
                "the hub reads an id in any case, so its cache does too");
        assertEquals(
                repo.resolve("snapshots/abc123def"),
                Hub.cachedPath(ModelRef.parse("hf.co/ggml-org/stories15M_MOE@abc123def"), hub),
                "a pinned commit needs no indirection");
        assertEquals(
                repo.resolve("snapshots/abc123def/stories15M_MOE-Q8_0.gguf"),
                Hub.cachedPath(
                        ModelRef.parse("hf.co/ggml-org/stories15M_MOE/stories15M_MOE-Q8_0.gguf"),
                        hub),
                "an explicit file resolves to itself");
        assertFalse(
                Files.exists(Hub.cachedPath(ModelRef.parse("hf.co/who/else"), hub)),
                "a repository that is not there is not a hit");
        assertNull(
                Hub.cachedPath(ModelRef.parse("modelscope.cn/ggml-org/stories15M_MOE"), hub),
                "ModelScope keeps its own layout; not ours to read");
    }

    @Test
    void linkPublishesABlobAsARelativeSymlink(@TempDir Path repo) throws IOException {
        Path blob = repo.resolve("blobs").resolve("a".repeat(64));
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, "weights");
        Path dest = repo.resolve("snapshots").resolve("c".repeat(40)).resolve("x.gguf");

        assertEquals(dest, Hub.link(blob, dest));
        assertTrue(Files.isSymbolicLink(dest));
        assertFalse(
                Files.readSymbolicLink(dest).isAbsolute(),
                "relative, so the cache can move as a whole");
        assertEquals("weights", Files.readString(dest));
        assertEquals(dest, Hub.link(blob, dest), "publishing twice is a no-op");
    }

    @Test
    void hubCacheGgufsAreListedAsRefs(@TempDir Path hub) throws IOException {
        String commit = "c".repeat(40);
        Path repo = hub.resolve("models--ggml-org--stories15M_MOE");
        Path snapshot = repo.resolve("snapshots").resolve(commit);
        Files.createDirectories(snapshot.resolve("sub"));
        Files.createDirectories(repo.resolve("refs"));
        Files.writeString(repo.resolve("refs/main"), commit);
        // as the hub lays it out: a snapshot entry is a relative link into blobs/
        Files.createDirectories(repo.resolve("blobs"));
        Files.writeString(repo.resolve("blobs").resolve("a".repeat(64)), "weights");
        Files.createSymbolicLink(
                snapshot.resolve("model-Q8_0.gguf"), Path.of("../../blobs/" + "a".repeat(64)));
        // a link whose blob is gone is not a model: skipped, quietly
        Files.createSymbolicLink(
                snapshot.resolve("stale.gguf"), Path.of("../../blobs/" + "b".repeat(64)));
        Files.writeString(snapshot.resolve("sub/mmproj-f16.gguf"), "projector");
        Files.writeString(snapshot.resolve("config.json"), "{}"); // the Python stack's litter
        // a snapshot refs/ no longer names is history, not the cache's current answer
        Files.createDirectories(repo.resolve("snapshots").resolve("d".repeat(40)));
        Files.writeString(
                repo.resolve("snapshots").resolve("d".repeat(40)).resolve("old.gguf"), "old");
        // a repository with no refs cannot say which snapshot is current: skipped, not guessed
        Files.createDirectories(hub.resolve("models--who--else/snapshots"));

        List<ModelStore.Cached> listed = Hub.cached(hub);
        assertEquals(
                List.of(
                        "hf.co/ggml-org/stories15M_MOE/model-Q8_0.gguf",
                        "hf.co/ggml-org/stories15M_MOE/sub/mmproj-f16.gguf"),
                listed.stream().map(ModelStore.Cached::ref).toList());
        assertEquals(7, listed.getFirst().sizeBytes(), "the blob's size, through the link");
    }

    @Test
    void everySnapshotARefNamesIsListedByThatRef(@TempDir Path hub) throws IOException {
        Path repo = hub.resolve("models--a--b");
        String main = "a".repeat(40), dev = "b".repeat(40), pr = "c".repeat(40);
        Hub.writeRef(repo.resolve("refs/main"), main);
        Hub.writeRef(repo.resolve("refs/v1"), main); // a tag on main's commit: listed once
        Hub.writeRef(repo.resolve("refs/dev"), dev);
        Hub.writeRef(repo.resolve("refs/pr/1"), pr);
        for (String commit : List.of(main, dev, pr)) {
            Path file = repo.resolve("snapshots/" + commit + "/m-Q8_0.gguf");
            Files.createDirectories(file.getParent());
            Files.writeString(file, "weights");
        }
        // no main: the branch there is still named, never passed off as the default
        Path other = hub.resolve("models--c--d");
        Hub.writeRef(other.resolve("refs/dev"), dev);
        Files.createDirectories(other.resolve("snapshots/" + dev));
        Files.writeString(other.resolve("snapshots/" + dev + "/m-Q8_0.gguf"), "weights");

        assertEquals(
                List.of(
                        "hf.co/a/b/m-Q8_0.gguf",
                        "hf.co/a/b@" + pr + "/m-Q8_0.gguf",
                        "hf.co/a/b@dev/m-Q8_0.gguf",
                        "hf.co/c/d@dev/m-Q8_0.gguf"),
                Hub.cached(hub).stream().map(ModelStore.Cached::ref).toList());
        assertEquals(List.of(), Hub.detached(hub), "a version a ref names is never pruned");
    }

    @Test
    void removingAVersionsLastFileTakesTheRefsNamingIt(@TempDir Path hub) throws IOException {
        Path repo = hub.resolve("models--a--b");
        String main = "a".repeat(40), dev = "b".repeat(40);
        Hub.writeRef(repo.resolve("refs/main"), main);
        Hub.writeRef(repo.resolve("refs/dev"), dev);
        Hub.writeRef(repo.resolve("refs/pr/1"), dev);
        Path onMain = repo.resolve("snapshots/" + main + "/m-Q8_0.gguf");
        Path onDev = repo.resolve("snapshots/" + dev + "/m-Q4_0.gguf");
        for (Path file : List.of(onMain, onDev)) {
            Files.createDirectories(file.getParent());
            Files.writeString(file, "weights");
        }

        assertTrue(Hub.remove(onDev, hub));
        assertFalse(Files.exists(repo.resolve("refs/dev")), "dev named only what is gone");
        assertFalse(Files.exists(repo.resolve("refs/pr")), "so did pr/1, and its folder");
        assertTrue(Files.exists(repo.resolve("refs/main")), "main still names its version");
        assertEquals(
                List.of("hf.co/a/b/m-Q8_0.gguf"),
                Hub.cached(hub).stream().map(ModelStore.Cached::ref).toList());

        assertTrue(Hub.remove(onMain, hub));
        assertFalse(Files.exists(repo), "nothing left: the repository folder goes");
    }

    /**
     * Where symlinks need a privilege (Windows outside Developer Mode) a snapshot entry is a hard
     * link: still its blob, so removing the last one frees it, and a named version's link counts.
     */
    @Test
    void aHardLinkedEntryIsItsBlob(@TempDir Path hub) throws IOException {
        Path repo = hub.resolve("models--a--b");
        String main = "a".repeat(40), old = "b".repeat(40);
        Hub.writeRef(repo.resolve("refs/main"), main);
        Path blob = repo.resolve("blobs").resolve("1".repeat(64));
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, "weights");
        Path current = repo.resolve("snapshots/" + main + "/m-Q8_0.gguf");
        Path replaced = repo.resolve("snapshots/" + old + "/m-Q8_0.gguf");
        for (Path entry : List.of(current, replaced)) {
            Files.createDirectories(entry.getParent());
            Files.createLink(entry, blob);
        }

        Hub.link(blob, current);
        assertFalse(Files.isSymbolicLink(current), "already the blob: kept, not redone");
        assertEquals(
                0,
                Hub.detached(hub).stream().mapToLong(ModelStore.Cached::sizeBytes).sum(),
                "main still holds those bytes");
        assertTrue(Hub.remove(replaced, hub));
        assertTrue(Files.exists(blob), "main's entry still links it");
        assertTrue(Hub.remove(current, hub));
        assertFalse(Files.exists(repo), "the last link took the blob, then the folder");
    }

    @Test
    void removeSparesABlobStillLinkedElsewhereThenTheFolder(@TempDir Path hub) throws IOException {
        Path repo = hub.resolve("models--a--b");
        Hub.writeRef(repo.resolve("refs/main"), "d".repeat(40));
        Path blob = repo.resolve("blobs").resolve("a".repeat(64));
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, "weights");
        Path first = repo.resolve("snapshots").resolve("c".repeat(40)).resolve("x.gguf");
        Path second = repo.resolve("snapshots").resolve("d".repeat(40)).resolve("x.gguf");
        Hub.link(blob, first);
        Hub.link(blob, second);

        assertTrue(Hub.remove(first, hub));
        assertTrue(Files.exists(blob), "another snapshot still links these bytes");
        assertFalse(Files.exists(first.getParent()), "an emptied snapshot folder goes");
        assertTrue(Hub.remove(second, hub));
        assertFalse(Files.exists(blob), "the last link took the blob with it");
        assertFalse(Files.exists(repo), "nothing left but refs: the repository folder goes");
        assertFalse(Hub.remove(second, hub), "the second remove is a miss");
    }

    @Test
    void detachedVersionsReportWhatRemovingThemFrees(@TempDir Path hub) throws IOException {
        Path repo = hub.resolve("models--a--b");
        Hub.writeRef(repo.resolve("refs/main"), "d".repeat(40));
        Path replaced = repo.resolve("blobs").resolve("1".repeat(64));
        Path kept = repo.resolve("blobs").resolve("2".repeat(64));
        Files.createDirectories(replaced.getParent());
        Files.writeString(replaced, "old!");
        Files.writeString(kept, "same");
        for (String commit : List.of("c".repeat(40), "e".repeat(40))) {
            Hub.link(replaced, repo.resolve("snapshots/" + commit + "/x-Q8_0.gguf"));
            Hub.link(kept, repo.resolve("snapshots/" + commit + "/y-Q8_0.gguf"));
        }
        Hub.link(kept, repo.resolve("snapshots/" + "d".repeat(40) + "/y-Q8_0.gguf"));

        List<ModelStore.Cached> detached = Hub.detached(hub);
        assertEquals(4, detached.size(), "both versions no ref names, every GGUF in them");
        assertEquals("hf.co/a/b@" + "c".repeat(40) + "/x-Q8_0.gguf", detached.getFirst().ref());
        assertEquals(
                4,
                detached.stream().mapToLong(ModelStore.Cached::sizeBytes).sum(),
                "the replaced blob once; the one main still links frees nothing");
    }

    @Test
    void aPlainUrlIsCachedUnderItsHostAndPath(@TempDir Path root) {
        System.setProperty("jinfer.models", root.toString());
        System.setProperty("jinfer.offline", "true"); // never fetch: we assert the PATH it picked
        try {
            var offline =
                    assertThrows(
                            ModelNotFoundException.class,
                            () ->
                                    ModelStore.standard()
                                            .resolve("https://example.org/models/x.gguf"));
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
            assertThrows(
                    IllegalArgumentException.class, () -> ModelStore.standard().resolve(bad), bad);
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
    void aDownloadThatCannotFitIsRefusedBeforeItStarts(@TempDir Path dir) throws IOException {
        var failure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                ModelStore.requireDiskSpace(
                                        dir.resolve("huge.gguf"), Long.MAX_VALUE / 2));
        String store = Files.getFileStore(dir) + " has ";
        assertTrue(failure.getMessage().contains(store), failure.getMessage());
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

    @Test
    void theBrowserSpellingOfAHostIsTheSameRef() {
        ModelRef ref = ModelRef.parse("huggingface.co/unsloth/Qwen3.5-4B-GGUF:Q8_0");
        assertEquals("hf.co", ref.host());
        assertEquals("unsloth", ref.owner());
        assertEquals("Q8_0", ref.quant());
        assertTrue(ModelRef.isRef("huggingface.co/unsloth/Qwen3.5-4B-GGUF"));
        assertEquals("modelscope.cn", ModelRef.parse("www.modelscope.cn/a/b").host());
    }
}
