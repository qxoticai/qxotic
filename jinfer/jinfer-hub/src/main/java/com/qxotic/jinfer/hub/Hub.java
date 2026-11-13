package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;

/**
 * The shared Hugging Face hub cache ({@code models--owner--repo/blobs/<sha256>} plus a {@code
 * snapshots/<commit>} symlink), as something jinfer READS and - on the default root - WRITES, so
 * the bytes are immediately visible to llama.cpp, {@code hf download} and everything else that
 * reads that layout, and theirs to us. Package-visible: the hub cache is an interop detail of the
 * store, not part of the API.
 */
final class Hub {

    private Hub() {}

    /** The exact requested path, before deciding whether it names a file or a folder. */
    static Path cachedPath(ModelRef ref, Path hubCache) throws IOException {
        if (!ref.host().equals(ModelRef.Host.HF.name)) {
            return null; // ModelScope's own cache is a different layout; not ours to read
        }
        Path repo = repoFolder(ref, hubCache).normalize();
        String revision = ref.revisionOrDefault();
        Path branch = repo.resolve("refs").resolve(revision);
        if (Files.isRegularFile(branch)) {
            revision = Files.readString(branch, StandardCharsets.UTF_8).strip();
        }
        Path snapshot = repo.resolve("snapshots").resolve(revision);
        return snapshot.resolve(ref.path());
    }

    /**
     * {@code ref}'s repository folder in the hub cache, in any case ({@link ModelStore#anyCase}).
     */
    static Path repoFolder(ModelRef ref, Path hubCache) {
        return ModelStore.anyCase(hubCache, "models--" + ref.owner() + "--" + ref.repo());
    }

    static Path cache() {
        for (String[] var : new String[][] {{"HF_HUB_CACHE", ""}, {"HF_HOME", "hub"}}) {
            String value = System.getenv(var[0]);
            if (value != null && !value.isBlank()) {
                return var[1].isEmpty() ? Path.of(value) : Path.of(value, var[1]);
            }
        }
        String xdg = System.getenv("XDG_CACHE_HOME");
        return xdg != null && !xdg.isBlank()
                ? Path.of(xdg, "huggingface", "hub")
                : Path.of(ModelStore.home(), ".cache", "huggingface", "hub");
    }

    /**
     * GGUFs in every snapshot a ref names, as refs that load them: {@code main}'s as {@code
     * hf.co/owner/repo/path}, another branch's or tag's as {@code hf.co/owner/repo@name/path}, and
     * one under a nested ref ({@code refs/pr/1}) by its commit. A commit two refs name is listed
     * once, {@code main} first.
     */
    static List<ModelStore.Cached> cached(Path hubCache) {
        List<ModelStore.Cached> found = new ArrayList<>();
        try {
            for (Path repo : repos(hubCache)) {
                Map<String, String> refs = refs(repo);
                Map<String, String> named = new LinkedHashMap<>(); // commit -> how a ref names it
                if (refs.containsKey("main")) named.put(refs.get("main"), "");
                refs.forEach(
                        (name, commit) ->
                                named.putIfAbsent(
                                        commit, "@" + (name.contains("/") ? commit : name)));
                for (var entry : named.entrySet()) {
                    Path snapshot = repo.resolve("snapshots").resolve(entry.getKey());
                    visitGgufs(
                            snapshot,
                            (file, size) ->
                                    found.add(
                                            new ModelStore.Cached(
                                                    ref(repo, entry.getValue(), snapshot, file),
                                                    size)));
                }
            }
        } catch (IOException e) {
            throw unreadable(hubCache, e);
        }
        found.sort(Comparator.comparing(ModelStore.Cached::ref)); // a walk's order is the disk's
        return found;
    }

    /**
     * GGUFs in snapshots no ref names: versions a pull replaced, or ones pinned to a commit, as
     * {@code hf.co/owner/repo@commit/path} refs. Each size is what removing the file frees: zero
     * when a named snapshot links the same blob, and a blob counts once, so the sizes add up to
     * what removing all of them frees.
     */
    static List<ModelStore.Cached> detached(Path hubCache) {
        List<ModelStore.Cached> found = new ArrayList<>();
        try {
            for (Path repo : repos(hubCache)) {
                Path snapshots = repo.resolve("snapshots");
                if (!Files.isDirectory(snapshots)) continue;
                Collection<String> named = refs(repo).values();
                List<Path> all;
                try (var listing = Files.list(snapshots)) {
                    all = listing.toList();
                }
                Set<Path> counted = new HashSet<>(); // what a named snapshot keeps is not freed
                for (Path snapshot : all) {
                    if (named.contains(snapshot.getFileName().toString())) {
                        ModelStore.visitFiles(
                                snapshot, (file, attrs) -> counted.add(identity(file, repo)));
                    }
                }
                for (Path snapshot : all) {
                    String commit = snapshot.getFileName().toString();
                    if (named.contains(commit)) continue;
                    visitGgufs(
                            snapshot,
                            (file, size) ->
                                    found.add(
                                            new ModelStore.Cached(
                                                    ref(repo, "@" + commit, snapshot, file),
                                                    counted.add(identity(file, repo)) ? size : 0)));
                }
            }
        } catch (IOException e) {
            throw unreadable(hubCache, e);
        }
        found.sort(Comparator.comparing(ModelStore.Cached::ref)); // a walk's order is the disk's
        return found;
    }

    /** The {@code models--owner--repo} folders of the hub cache. */
    private static List<Path> repos(Path hubCache) throws IOException {
        if (!Files.isDirectory(hubCache)) return List.of();
        try (var folders = Files.list(hubCache)) {
            return folders.filter(f -> Files.isDirectory(f) && repoId(f) != null).toList();
        }
    }

    /** {@code owner/repo} of a {@code models--owner--repo} folder, or null for any other. */
    private static String repoId(Path repo) {
        String folder = repo.getFileName().toString();
        int cut = folder.indexOf("--", "models--".length());
        if (!folder.startsWith("models--") || cut < 0) {
            return null;
        }
        return folder.substring("models--".length(), cut) + "/" + folder.substring(cut + 2);
    }

    /** Each ref under {@code refs/} (a branch, a tag, {@code pr/1}) and its commit, by name. */
    private static Map<String, String> refs(Path repo) throws IOException {
        Map<String, String> refs = new TreeMap<>();
        Path dir = repo.resolve("refs");
        if (Files.isDirectory(dir)) {
            ModelStore.visitFiles(
                    dir,
                    (file, attrs) -> {
                        try {
                            String commit = Files.readString(file, StandardCharsets.UTF_8).strip();
                            String name = dir.relativize(file).toString().replace('\\', '/');
                            if (isCommit(commit)) refs.put(name, commit);
                        } catch (IOException unreadable) {
                            // a ref no one can read names nothing
                        }
                    });
        }
        return refs;
    }

    /** Each GGUF under {@code snapshot}, with its size (the blob's, through the link). */
    private static void visitGgufs(Path snapshot, BiConsumer<Path, Long> onGguf)
            throws IOException {
        if (!Files.isDirectory(snapshot)) return;
        ModelStore.visitFiles(
                snapshot,
                (file, attrs) -> {
                    if (ModelStore.isGguf(file.getFileName().toString())) {
                        onGguf.accept(file, attrs.size());
                    }
                });
    }

    /** {@code hf.co/owner/repo<at>/path} for {@code file} under {@code snapshot}. */
    private static String ref(Path repo, String at, Path snapshot, Path file) {
        StringBuilder ref = new StringBuilder("hf.co/").append(repoId(repo)).append(at);
        for (Path segment : snapshot.relativize(file)) {
            ref.append('/').append(segment);
        }
        return ref.toString();
    }

    private static UncheckedIOException unreadable(Path hubCache, IOException e) {
        return new UncheckedIOException(
                "could not read the Hugging Face hub cache at "
                        + hubCache
                        + ": "
                        + ModelStore.why(e),
                e);
    }

    /**
     * The blob a snapshot entry is: a symlink's target, or the file in {@code blobs/} a hard link
     * shares; null for a copy, which cannot tell.
     */
    private static Path blobOf(Path entry, Path repo) throws IOException {
        if (Files.isSymbolicLink(entry)) return target(entry);
        Path blobs = repo.resolve("blobs");
        if (!Files.isRegularFile(entry) || !Files.isDirectory(blobs)) return null;
        try (var all = Files.list(blobs)) {
            for (Path blob : all.toList()) {
                if (Files.isSameFile(entry, blob)) return blob.toAbsolutePath().normalize();
            }
        }
        return null;
    }

    /** The bytes a snapshot entry holds, as one key per blob: the entry itself for a copy. */
    private static Path identity(Path entry, Path repo) {
        try {
            Path blob = blobOf(entry, repo);
            return blob != null ? blob : entry;
        } catch (IOException unreadable) {
            return entry;
        }
    }

    /** What a symbolic link points to, as an absolute path. */
    private static Path target(Path link) throws IOException {
        return link.getParent().resolve(Files.readSymbolicLink(link)).toAbsolutePath().normalize();
    }

    /**
     * Removes one snapshot entry from the hub cache, and its blob when this was the last snapshot
     * linking it - the same care llama.cpp takes, because a blob may serve many snapshots. A ref
     * left naming an emptied snapshot goes with it, and once the repository holds no file but its
     * refs, the whole folder goes, as {@code hf cache delete} leaves it.
     */
    static boolean remove(Path cached, Path hubCache) throws IOException {
        Path repo = hubCache.resolve(hubCache.relativize(cached).getName(0));
        Path blob = blobOf(cached, repo);
        if (!Files.deleteIfExists(cached)) return false;
        if (blob != null && Files.isRegularFile(blob) && !referenced(blob)) {
            Files.deleteIfExists(blob);
        }
        ModelStore.pruneEmpty(cached.getParent(), repo);
        ModelStore.pruneEmpty(repo.resolve("blobs"), repo);
        // a version left with no file: a ref naming it would name nothing
        Path snapshot = repo.resolve(repo.relativize(cached).subpath(0, 2));
        if (Files.notExists(snapshot)) {
            for (var ref : refs(repo).entrySet()) {
                if (ref.getValue().equals(snapshot.getFileName().toString())) {
                    Path file = repo.resolve("refs").resolve(ref.getKey());
                    Files.deleteIfExists(file);
                    ModelStore.pruneEmpty(file.getParent(), repo);
                }
            }
        }
        try (var children = Files.list(repo)) {
            if (children.anyMatch(child -> !child.getFileName().toString().equals("refs")))
                return true;
        }
        try (var refs = Files.walk(repo)) {
            for (Path path : refs.sorted(Comparator.reverseOrder()).toList())
                Files.deleteIfExists(path);
        }
        return true;
    }

    /** Whether any snapshot entry in the blob's repository still links {@code blob}. */
    private static boolean referenced(Path blob) throws IOException {
        Path snapshots = blob.getParent().getParent().resolve("snapshots");
        if (!Files.isDirectory(snapshots)) {
            return false;
        }
        try (var walk = Files.walk(snapshots)) {
            for (Path entry : walk.toList()) {
                if (Files.isRegularFile(entry)
                        && Files.isSameFile(entry, blob)) { // either link kind
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Downloads {@code file} into the hub cache the way every HF client lays it out: bytes in
     * {@code blobs/<sha256>} - the same sha256 the download already verifies - and a relative
     * symlink from {@code snapshots/<commit>/<path>}, so llama.cpp and {@code hf download} see the
     * file as their own. The store serializes revision selection and publication; payload
     * replacement also holds its destination lock. Other clients publish content-addressed blobs
     * independently, so their complete bytes at the same hash are interchangeable. The caller holds
     * the publication-key lock before resolving the revision.
     */
    static Path fetchInto(
            ModelRef ref,
            RemoteFile file,
            String commit,
            Path hubCache,
            RepositorySource source,
            boolean force)
            throws IOException {
        Path repo = repoFolder(ref, hubCache);
        Path blob = repo.resolve("blobs").resolve(file.sha256());
        Path dest = ModelStore.under(repo.resolve("snapshots").resolve(commit), file.path());
        // a blob is named by its sha256 once verified (by jinfer) or once its size checks (other HF
        // clients): name and size say it is current, as every HF client takes them, without a
        // read. Repairing bytes that rotted after publication is what force is for.
        boolean reuse =
                !force
                        && Files.isRegularFile(blob)
                        && (file.sizeBytes() < 0 || Files.size(blob) == file.sizeBytes());
        if (!reuse) {
            ModelStore.requireDiskSpace(
                    blob, Fetch.replacementRemainingBytes(blob, file.sizeBytes()));
            Fetch.announce("download " + ref.host() + "/" + ref.repoId() + "/" + file.path());
            // Pin the same commit and source used by the listing, including configured mirrors.
            String url = source.fileUrl(ref, commit, file.path());
            Fetch.replace(
                    blob,
                    file.sizeBytes(),
                    null, // Fetch.download verifies the sha256; a second read would only repeat it
                    stage ->
                            Fetch.download(
                                    url,
                                    stage,
                                    ModelStore.nameOf(file.path()),
                                    file.sizeBytes(),
                                    file.sha256(),
                                    RepositorySource.headers(ModelRef.Host.HF)));
        }
        Path published = link(blob, dest);
        if (!isCommit(ref.revisionOrDefault())) {
            writeRef(repo.resolve("refs").resolve(ref.revisionOrDefault()), commit);
        }
        if (reuse) Fetch.announce("up to date: " + ref);
        return published;
    }

    static Path publicationKey(ModelRef ref, Path hubCache) {
        return repoFolder(ref, hubCache).resolve("refs").resolve(ref.revisionOrDefault());
    }

    /**
     * Publishes {@code blob} at {@code dest} as a relative symlink, the hub cache's own idiom.
     * Where symlinks are refused (Windows outside Developer Mode) the entry is a hard link, and a
     * copy only where hard links are refused too. An entry already linking the blob is kept. Never
     * move a shared blob out from under another snapshot's link. Publish before changing the branch
     * ref (PULL-REFRESH.md).
     */
    static Path link(Path blob, Path dest) throws IOException {
        if (Files.exists(dest) && Files.isSameFile(dest, blob)) {
            return dest; // a link to it, either kind: nothing to redo, nothing a reader holds moved
        }
        Files.createDirectories(dest.getParent());
        Path tmp = staging(dest.getParent(), ".jinfer-link-");
        try {
            Path target = dest.getParent().toAbsolutePath().relativize(blob.toAbsolutePath());
            Files.deleteIfExists(tmp); // a stale one would be in the link's way
            try {
                Files.createSymbolicLink(tmp, target);
            } catch (IOException | UnsupportedOperationException noSymlinks) {
                try { // Windows outside Developer Mode: a hard link needs no privilege
                    Files.createLink(tmp, blob);
                } catch (IOException | UnsupportedOperationException noHardLinks) {
                    Files.copy(blob, tmp, StandardCopyOption.REPLACE_EXISTING);
                }
            }
            Fetch.publish(tmp, dest);
        } finally {
            Fetch.cleanup(tmp);
        }
        return dest;
    }

    /**
     * Writes a {@code refs/<branch>} file the way the hub does: content is the commit, atomically.
     */
    static void writeRef(Path refFile, String commit) throws IOException {
        byte[] bytes = commit.getBytes(StandardCharsets.UTF_8);
        if (Files.isRegularFile(refFile) && Arrays.equals(Files.readAllBytes(refFile), bytes))
            return;
        Files.createDirectories(refFile.getParent());
        Path tmp = staging(refFile.getParent(), ".jinfer-ref-");
        try {
            Files.write(tmp, bytes);
            Fetch.publish(tmp, refFile);
        } finally {
            Fetch.cleanup(tmp);
        }
    }

    private static final AtomicLong STAGED = new AtomicLong();

    /**
     * A {@code .part} name of the folder for this process to write: pid and count, so only a stale
     * file of an earlier process can hold it, and that one is overwritten. No random name: a
     * SecureRandom is the security providers, in a native image. The pid is read at use, as a
     * static would be the image builder's.
     */
    private static Path staging(Path dir, String prefix) {
        return dir.resolve(
                prefix + ProcessHandle.current().pid() + "-" + STAGED.incrementAndGet() + ".part");
    }

    /** A git commit hash, fit to be a snapshot directory name. */
    static boolean isCommit(String value) {
        return isHex(value, 40);
    }

    /** A sha256, fit to be a blob file name. The oid arrives from a remote listing: validated. */
    static boolean isSha256(String value) {
        return isHex(value, 64);
    }

    private static boolean isHex(String value, int length) {
        return value != null && value.matches("[0-9a-f]{" + length + "}");
    }
}
