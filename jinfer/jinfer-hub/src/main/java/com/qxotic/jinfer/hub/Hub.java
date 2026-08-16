package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/**
 * The shared HuggingFace hub cache ({@code models--owner--repo/blobs/<sha256>} plus a {@code
 * snapshots/<commit>} symlink), as something jinfer READS and - on the default root - WRITES, so
 * the bytes are immediately visible to llama.cpp, {@code hf download} and everything else that
 * reads that layout, and theirs to us. Package-visible: the hub cache is an interop detail of the
 * store, not part of the API.
 */
final class Hub {

    private Hub() {}

    /**
     * The snapshot directory this ref names in the hub cache, or null. Honors the same variables
     * every HF client reads, and resolves a branch through {@code refs/} exactly as they do - a ref
     * pinned to a commit needs no indirection at all.
     */
    static Path snapshot(ModelRef ref) throws IOException {
        return snapshot(ref, cache());
    }

    /** Package-visible for its test: the env lookup is the only part a test cannot drive. */
    static Path snapshot(ModelRef ref, Path hubCache) throws IOException {
        if (!ref.host().equals(ModelRef.Host.HF.name)) {
            return null; // ModelScope's own cache is a different layout; not ours to read
        }
        Path repo = hubCache.resolve("models--" + ref.owner() + "--" + ref.repo()).normalize();
        String revision = ref.revisionOrDefault();
        Path branch = repo.resolve("refs").resolve(revision);
        if (Files.isRegularFile(branch)) {
            revision = Files.readString(branch, StandardCharsets.UTF_8).strip();
        }
        Path snapshot = repo.resolve("snapshots").resolve(revision);
        Path located = snapshot.resolve(ref.path());
        if (Files.isRegularFile(located)) return located.getParent();
        return Files.isDirectory(located) ? located : null;
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
                : Path.of(System.getProperty("user.home"), ".cache", "huggingface", "hub");
    }

    /** GGUFs in the hub cache's CURRENT snapshots, as refs. */
    static List<ModelStore.Cached> cached(Path hubCache) {
        if (!Files.isDirectory(hubCache)) {
            return List.of();
        }
        List<ModelStore.Cached> refs = new ArrayList<>();
        try (var repos = Files.list(hubCache)) {
            for (Path repo : repos.filter(Files::isDirectory).toList()) {
                String folder = repo.getFileName().toString();
                int cut = folder.indexOf("--", "models--".length());
                if (!folder.startsWith("models--") || cut < 0) {
                    continue;
                }
                String repoId =
                        folder.substring("models--".length(), cut)
                                + "/"
                                + folder.substring(cut + 2);
                String commit = currentCommit(repo);
                Path snapshot = commit == null ? null : repo.resolve("snapshots").resolve(commit);
                if (snapshot == null || !Files.isDirectory(snapshot)) {
                    continue;
                }
                try (var walk = Files.walk(snapshot)) {
                    for (Path file :
                            walk.filter(Files::isRegularFile) // a broken symlink cannot resolve
                                    .filter(p -> ModelStore.isGguf(p.getFileName().toString()))
                                    .sorted()
                                    .toList()) {
                        StringBuilder ref = new StringBuilder("hf.co/").append(repoId);
                        for (Path segment : snapshot.relativize(file)) {
                            ref.append('/').append(segment);
                        }
                        refs.add(new ModelStore.Cached(ref.toString(), ModelStore.sizeOf(file)));
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "could not read the HuggingFace hub cache at " + hubCache + ": " + e, e);
        }
        return refs;
    }

    /** The commit a repo's {@code refs/} names right now: {@code main} first, any branch else. */
    private static String currentCommit(Path repo) throws IOException {
        Path refs = repo.resolve("refs");
        if (!Files.isDirectory(refs)) {
            return null;
        }
        String fallback = null;
        try (var entries = Files.list(refs)) {
            for (Path entry : entries.filter(Files::isRegularFile).sorted().toList()) {
                String commit = Files.readString(entry, StandardCharsets.UTF_8).strip();
                if (!isCommit(commit)) {
                    continue;
                }
                if (entry.getFileName().toString().equals("main")) {
                    return commit;
                }
                if (fallback == null) {
                    fallback = commit;
                }
            }
        }
        return fallback;
    }

    /**
     * Removes one snapshot entry from the hub cache, and its blob when this was the last snapshot
     * linking it - the same care llama.cpp takes, because a blob may serve many snapshots.
     * Package-visible for its test.
     */
    static boolean evict(Path cached) throws IOException {
        Path blob =
                Files.isSymbolicLink(cached)
                        ? cached.getParent()
                                .resolve(Files.readSymbolicLink(cached))
                                .toAbsolutePath()
                                .normalize()
                        : null;
        boolean removed = Files.deleteIfExists(cached);
        if (removed && blob != null && Files.isRegularFile(blob) && !referenced(blob)) {
            Files.deleteIfExists(blob);
        }
        return removed;
    }

    /** Whether any snapshot entry in the blob's repository still links {@code blob}. */
    private static boolean referenced(Path blob) throws IOException {
        Path snapshots = blob.getParent().getParent().resolve("snapshots");
        if (!Files.isDirectory(snapshots)) {
            return false;
        }
        try (var walk = Files.walk(snapshots)) {
            for (Path entry : walk.toList()) {
                if (Files.isSymbolicLink(entry)
                        && entry.getParent()
                                .resolve(Files.readSymbolicLink(entry))
                                .toAbsolutePath()
                                .normalize()
                                .equals(blob)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * The commit this ref's revision names, or null when it cannot be learned - and null only
     * DOWNGRADES a download to the flat layout, never fails it. The hub layout keys snapshots by
     * commit, so joining it starts here; a ref already pinned to a commit needs no request.
     */
    static String commit(ModelRef ref) {
        String revision = ref.revisionOrDefault();
        return isCommit(revision)
                ? revision
                : new RepositorySource(ModelRef.Host.HF, null).commitFor(ref);
    }

    /**
     * Downloads {@code file} into the hub cache the way every HF client lays it out: bytes in
     * {@code blobs/<sha256>} - the same sha256 the download already verifies - and a relative
     * symlink from {@code snapshots/<commit>/<path>}, so llama.cpp and {@code hf download} see the
     * file as their own. Concurrent writers need no lock here: blobs are content-addressed, so two
     * tools racing on one file write identical bytes, each behind its own temp-and-rename.
     */
    static Path fetchInto(ModelRef ref, RemoteFile file, String commit, Path hubCache)
            throws IOException {
        Path repo = hubCache.resolve("models--" + ref.owner() + "--" + ref.repo());
        Path blob = repo.resolve("blobs").resolve(file.sha256());
        Path dest = ModelStore.under(repo.resolve("snapshots").resolve(commit), file.path());
        if (!Files.isRegularFile(blob)) {
            ModelStore.requireDiskSpace(blob, Fetch.remainingBytes(blob, file.sizeBytes()));
            Fetch.announce("download " + ref.host() + "/" + ref.repoId() + "/" + file.path());
            // download at the COMMIT, not the branch: the listing that chose this file and the
            // fetch must not straddle a push
            String url = ref.repoUrl() + "/resolve/" + commit + "/" + file.path();
            Fetch.download(
                    url,
                    blob,
                    ModelStore.nameOf(file.path()),
                    file.sizeBytes(),
                    file.sha256(),
                    RepositorySource.headers(ModelRef.Host.HF));
        }
        if (!isCommit(ref.revisionOrDefault())) {
            writeRef(repo.resolve("refs").resolve(ref.revisionOrDefault()), commit);
        }
        return link(blob, dest);
    }

    /**
     * Publishes {@code blob} at {@code dest} as a relative symlink, the hub cache's own idiom. On a
     * filesystem that refuses symlinks (Windows without developer mode) the blob MOVES to {@code
     * dest} instead - dedup is lost, correctness is not, and that is the same degraded mode
     * llama.cpp and huggingface_hub fall back to.
     */
    static Path link(Path blob, Path dest) throws IOException {
        if (Files.isSymbolicLink(dest) || Files.exists(dest)) {
            return dest; // the blob it points at was just ensured
        }
        Files.createDirectories(dest.getParent());
        try {
            Path target = dest.getParent().toAbsolutePath().relativize(blob.toAbsolutePath());
            Files.createSymbolicLink(dest, target);
        } catch (IOException | UnsupportedOperationException noSymlinks) {
            Files.move(blob, dest);
        }
        return dest;
    }

    /**
     * Writes a {@code refs/<branch>} file the way the hub does: content is the commit, atomically.
     */
    private static void writeRef(Path refFile, String commit) throws IOException {
        Files.createDirectories(refFile.getParent());
        Path tmp = refFile.resolveSibling(refFile.getFileName() + ".tmp");
        Files.writeString(tmp, commit, StandardCharsets.UTF_8);
        Files.move(
                tmp, refFile, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
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
        if (value == null || value.length() != length) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            boolean hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            if (!hex) {
                return false;
            }
        }
        return true;
    }
}
