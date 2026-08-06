package com.qxotic.jinfer.hub;

import com.qxotic.format.json.Json;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * The model cache: a {@link ModelRef} in, a local path out, downloading only what is missing.
 *
 * <p>THE CACHE IS THE REF. {@code hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0} lands at {@code
 * <root>/hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf}: subfolders are preserved, and
 * a named revision joins the repository directory as {@code repo@revision}. The mapping is
 * injective and reversible, so a path tells you where it came from and a ref tells you where it
 * will land. It is also exactly the tree {@code scripts/download-models.sh} populates and the test
 * {@code ModelFixture} reads, so a checkout that has downloaded its fixtures is already a warm
 * cache. Flat and obvious on purpose: {@code ls} and {@code rm -rf} are the management commands,
 * which is why this module ships no {@code list} or {@code rm}.
 *
 * <p>Root: {@code -Djinfer.models} &gt; {@code $JINFER_MODELS} &gt; the platform's cache directory
 * ({@code $XDG_CACHE_HOME} or {@code ~/.cache} on Linux, {@code ~/Library/Caches} on macOS, {@code
 * %LOCALAPPDATA%} on Windows) plus {@code jinfer}. Platform-native rather than {@code ~/.cache}
 * everywhere, because a cache in the wrong place is one the operating system will not clean up and
 * the user will not find.
 *
 * <p>Format policy lives HERE, not in the grammar. {@link ModelRef} parses any repository path;
 * this class knows that jinfer loads GGUF, and refuses a repository that ships none BEFORE any
 * bytes move rather than after a caller waits for twenty gigabytes.
 *
 * <p>Nothing in the inference engine calls this. Resolution happens in a CLI, before {@code
 * Models.load}, so a Java caller that passes a path gets exactly that path and no library load ever
 * touches the network.
 */
public final class ModelStore {

    private ModelStore() {}

    /**
     * A file as the repository lists it. {@code path} is REPOSITORY-RELATIVE, so it carries any
     * subfolder; {@code sha256} is null when the host does not publish one.
     */
    private record RepoFile(String path, long size, String sha256) {}

    /** A listing entry that is a directory: enough to tell a folder from a file. */
    private record RepoDir(String path) {}

    private static final Pattern SPLIT_PART =
            Pattern.compile(".*-\\d{5}-of-\\d{5}\\.gguf$", Pattern.CASE_INSENSITIVE);

    // ---- the cache root ----

    /** Where downloads land, and the first place a ref is looked for. */
    public static Path root() {
        String property = System.getProperty("jinfer.models");
        if (property != null && !property.isBlank()) {
            return Path.of(property);
        }
        String env = System.getenv("JINFER_MODELS");
        if (env != null && !env.isBlank()) {
            return Path.of(env);
        }
        return platformCache().resolve("jinfer");
    }

    private static Path platformCache() {
        String home = System.getProperty("user.home");
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("win")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            return localAppData != null && !localAppData.isBlank()
                    ? Path.of(localAppData)
                    : Path.of(home, "AppData", "Local");
        }
        if (os.contains("mac") || os.contains("darwin")) {
            return Path.of(home, "Library", "Caches");
        }
        String xdg = System.getenv("XDG_CACHE_HOME");
        return xdg != null && !xdg.isBlank() ? Path.of(xdg) : Path.of(home, ".cache");
    }

    /**
     * Marks the root per the Cache Directory Tagging Standard, so backup tools (borg, restic,
     * rsync) skip it. Model weights are re-downloadable by definition and nobody wants hundreds of
     * gigabytes of them in a backup. Best effort - a cache that cannot be tagged still works.
     */
    private static void tagCacheDirectory(Path root) {
        Path tag = root.resolve("CACHEDIR.TAG");
        try {
            if (!Files.exists(tag)) {
                Files.createDirectories(root);
                Files.writeString(
                        tag,
                        "Signature: 8a477f597d28d172789f06886806bc55\n"
                                + "# This file marks this directory as jinfer's model cache.\n"
                                + "# Its contents can be re-downloaded and need not be backed"
                                + " up.\n",
                        StandardCharsets.UTF_8);
            }
        } catch (IOException ignored) {
            // the tag is a courtesy to backup tools, never a precondition for a download
        }
    }

    // ---- the API ----

    /**
     * A local path or a remote ref, told apart by ONE visible rule: a ref names its host,
     * everything else is a file on this machine. Nothing is inferred from the shape of the string
     * or from what happens to exist on disk, so the same argument means the same thing on every
     * machine.
     */
    public static Path resolve(String pathOrRef) {
        if (ModelRef.isRef(pathOrRef)) {
            return resolve(ModelRef.parse(pathOrRef)); // a repository we know how to talk to
        }
        if (ModelRef.hostOfUrl(pathOrRef) != null) {
            return url(pathOrRef); // any other URL: bytes, and nothing else
        }
        Path local = localFile(pathOrRef);
        if (local == null) {
            throw new IllegalArgumentException(unresolvable(pathOrRef));
        }
        return local;
    }

    /**
     * A plain URL: fetched and cached, with none of what a repository gives us. There is no listing
     * to search, so no quant and no revision, and no published checksum, so the only integrity
     * check is the size the server states. That downgrade is announced rather than assumed - a
     * caller who thinks jinfer verified these bytes has been misled.
     *
     * <p>The cache mapping still holds: {@code https://example.org/models/x.gguf} lands at {@code
     * <root>/example.org/models/x.gguf}, so a path still tells you where it came from. The URL is
     * used VERBATIM for the request - a query may be a signature - while the cache path is derived
     * from the path portion alone.
     */
    private static Path url(String url) {
        java.net.URI uri;
        try {
            uri = java.net.URI.create(url);
        } catch (IllegalArgumentException malformed) {
            throw new IllegalArgumentException(
                    "not a URL: '" + url + "': " + malformed.getMessage());
        }
        String scheme = uri.getScheme() == null ? "" : uri.getScheme().toLowerCase(Locale.ROOT);
        require(
                scheme.equals("https") || scheme.equals("http"),
                "cannot fetch '" + url + "': only http and https URLs are supported");
        String path = uri.getPath() == null ? "" : uri.getPath();
        require(
                !path.isEmpty() && !path.endsWith("/") && !nameOf(path).isEmpty(),
                "the URL must end in the file name, so the cache has something to call it: " + url);

        Path dest = root().resolve(uri.getHost());
        for (String segment : path.split("/")) {
            if (!segment.isEmpty()) {
                require(
                        !segment.equals(".") && !segment.equals("..") && segment.indexOf('\\') < 0,
                        "the URL would escape the cache: " + url);
                dest = dest.resolve(segment);
            }
        }
        if (Files.isRegularFile(dest)) {
            return dest;
        }
        Map<String, String> headers = Map.of("User-Agent", "jinfer-hub");
        long size = Fetch.sizeOf(url, headers);
        requireOnlineFor(url, dest);
        requireDiskSpace(dest, Fetch.remainingBytes(dest, size));
        tagCacheDirectory(root());
        Fetch.progress()
                .println(
                        "download "
                                + uri.getHost()
                                + path
                                + "\n  "
                                + uri.getHost()
                                + " publishes no checksum - verifying size only");
        try {
            Fetch.download(url, dest, size, null, headers);
        } catch (IOException e) {
            throw new UncheckedIOException("could not fetch " + url + ": " + e, e);
        }
        return dest;
    }

    private static void requireOnlineFor(String what, Path dest) {
        if ("1".equals(System.getenv("JINFER_OFFLINE")) || Boolean.getBoolean("jinfer.offline")) {
            throw new IllegalStateException(
                    what + " is not cached at " + dest + " and JINFER_OFFLINE forbids downloading");
        }
    }

    /**
     * Everything in the cache, named the way you would ASK for it again - the inverse of
     * resolution, and the one thing a shell cannot do here: {@code find} knows the paths, but only
     * this knows which of them are refs.
     *
     * <p>A file under a known host is returned as its REF, rendered with {@code /} whatever the
     * platform separator is. Anything else - a tree predating the hub, a file fetched from a plain
     * URL whose scheme we did not record - is returned as its absolute PATH. Both forms are valid
     * {@code --model} arguments, and neither is guessed: this never invents a ref for a file that
     * has none. Sorted, with the cache's own scaffolding left out.
     */
    public static List<String> cached() {
        Path root = root();
        if (!Files.isDirectory(root)) {
            return List.of();
        }
        try (var walk = Files.walk(root)) {
            return walk.filter(Files::isRegularFile)
                    .map(root::relativize)
                    .filter(ModelStore::isCachedModel)
                    .map(relative -> asRefOrPath(root, relative))
                    .sorted()
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "could not read the model cache at " + root + ": " + e, e);
        }
    }

    /** A ref when the first segment is a host we know, else the absolute path. Never a guess. */
    private static String asRefOrPath(Path root, Path relative) {
        if (!ModelRef.isKnownHost(relative.getName(0).toString())) {
            return root.resolve(relative).toString();
        }
        StringBuilder ref = new StringBuilder();
        for (Path segment : relative) {
            if (!ref.isEmpty()) {
                ref.append('/');
            }
            ref.append(segment);
        }
        return ref.toString();
    }

    /** A cached model, as opposed to the cache's own bookkeeping. */
    private static boolean isCachedModel(Path relative) {
        String name = relative.getFileName().toString();
        if (relative.getName(0).toString().equals(".locks") || name.equals("CACHEDIR.TAG")) {
            return false;
        }
        return !name.endsWith(".part") && !name.endsWith(".map") && !name.endsWith(".lock");
    }

    /**
     * Forgets the cached file a ref resolves to, so the next resolve downloads it again. What
     * {@code pull --force} is for: this cache is pinned by name and never revalidates, which is
     * what makes a warm resolve cost zero requests, so a repository that re-uploads a quant under
     * the same name needs one way to say "fetch it again". A ref pinned to a commit never needs
     * this, because a commit is immutable.
     */
    public static boolean evict(String pathOrRef) {
        if (!ModelRef.isRef(pathOrRef)) {
            return false; // a file you passed by path is yours; this cache never deletes it
        }
        ModelRef ref = ModelRef.parse(pathOrRef);
        try {
            Path cached = cachedFile(ref);
            // only ever delete OUR cache: a hit may come from the HuggingFace hub cache, which
            // belongs to another tool and to every other model that shares its blobs
            if (cached == null || !cached.toAbsolutePath().startsWith(root().toAbsolutePath())) {
                return false;
            }
            return Files.deleteIfExists(cached);
        } catch (IOException e) {
            throw new UncheckedIOException("could not evict " + ref + ": " + e, e);
        }
    }

    /**
     * Why this string is neither a ref nor a file. A URL naming an unknown host gets its own
     * answer: it plainly meant to be remote, and telling someone to "name its host" when they just
     * did is the kind of message that makes people doubt the tool rather than the argument.
     */
    private static String unresolvable(String pathOrRef) {
        return "no such model file: '"
                + pathOrRef
                + "'. A REMOTE model names its host, for example"
                + " hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M or"
                + " modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0";
    }

    /**
     * The file this string names, or null when it names none.
     *
     * <p>A model and a companion are each ONE FILE. A directory is refused by name rather than
     * reported as missing, because "no such file" about a path that plainly exists sends people
     * looking in the wrong place.
     *
     * <p>WINDOWS: a string that is not a valid path there throws {@link
     * java.nio.file.InvalidPathException} instead of reporting a missing file. By the time we look,
     * a path is the only reading left, so one that cannot be parsed is simply not a file.
     */
    // ponytail: single files only. A directory-shaped companion (a LoRA with its config, a
    // converted checkpoint) needs a subtree download - the snapshot() this module deliberately
    // does not have. Relax this the day one exists, not before.
    private static Path localFile(String path) {
        try {
            Path candidate = Path.of(path);
            if (Files.isDirectory(candidate)) {
                throw new IllegalArgumentException(
                        "'" + path + "' is a directory; a model or companion is a single file");
            }
            return Files.isRegularFile(candidate) ? candidate : null;
        } catch (java.nio.file.InvalidPathException notAPath) {
            return null;
        }
    }

    // ---- resolution ----

    /** The cached file for {@code ref}, downloading it when absent. */
    static Path resolve(ModelRef ref) {
        try {
            Path cached = cachedFile(ref);
            return cached != null ? cached : fetch(ref, select(ref));
        } catch (IOException e) {
            // the cause carries the only actionable part (refused proxy, DNS, TLS, timeout)
            throw new UncheckedIOException("could not resolve " + ref + ": " + e, e);
        }
    }

    /**
     * The one file {@code ref} selects, from the repository listing.
     *
     * <p>The listing - never a file extension - decides whether the ref's path names a file or a
     * folder, which is what keeps the grammar free of formats: {@code .safetensors} and a folder
     * called {@code Qwen2.5} both work without a special case.
     */
    private static RepoFile select(ModelRef ref) throws IOException {
        String folder = ref.location();
        if (!folder.isEmpty()) {
            String parent = parentOf(folder);
            String last = nameOf(folder);
            List<RepoFile> files = listFiles(ref, parent);
            for (RepoFile file : files) {
                if (nameOf(file.path()).equals(last)) {
                    require(
                            ref.quant() == null,
                            ref.location()
                                    + " already names a file, so ':"
                                    + ref.quant()
                                    + "' has nothing to choose - drop one of them");
                    return file;
                }
            }
            if (!hasDirectory(ref, parent, last)) {
                throw new IllegalArgumentException(
                        "no '"
                                + last
                                + "' in "
                                + ref.repoId()
                                + (parent.isEmpty() ? "" : "/" + parent)
                                + ". Contains: "
                                + names(files));
            }
        }
        return byQuant(ref, folder);
    }

    /**
     * The single file matching this ref's quant inside {@code folder}, or a message saying why not.
     */
    private static RepoFile byQuant(ModelRef ref, String folder) throws IOException {
        List<RepoFile> files = listFiles(ref, folder);
        List<RepoFile> models = files.stream().filter(f -> isModelGguf(nameOf(f.path()))).toList();
        if (models.isEmpty()) {
            throw noGguf(ref, folder, files);
        }
        // one GGUF and no quant asked for: that is the model, whatever it is called. An EXPLICIT
        // quant never falls back - a caller who named one and got another has been lied to.
        if (models.size() == 1 && ref.quant() == null) {
            return models.get(0);
        }
        List<RepoFile> matches =
                models.stream()
                        .filter(f -> matchesQuant(nameOf(f.path()), ref.quantOrDefault()))
                        .toList();
        if (matches.size() == 1) {
            return matches.get(0);
        }
        if (matches.isEmpty()) {
            throw new IllegalArgumentException(
                    "no "
                            + ref.quantOrDefault()
                            + " in "
                            + ref.repoId()
                            + (folder.isEmpty() ? "" : "/" + folder)
                            + ". Available: "
                            + names(models));
        }
        throw new IllegalArgumentException(
                ref.quantOrDefault()
                        + " matches "
                        + matches.size()
                        + " files in "
                        + ref.repoId()
                        + (folder.isEmpty() ? "" : "/" + folder)
                        + " - name the one you want:"
                        + menu(ref, matches));
    }

    /**
     * A repository with no GGUF in it. Refused HERE, before a download, because the alternative is
     * a caller waiting for twenty gigabytes of safetensors and then being told jinfer cannot open
     * them.
     */
    private static IllegalArgumentException noGguf(
            ModelRef ref, String folder, List<RepoFile> files) {
        boolean safetensors =
                files.stream()
                        .anyMatch(
                                f ->
                                        nameOf(f.path())
                                                .toLowerCase(Locale.ROOT)
                                                .endsWith(".safetensors"));
        String where = ref.repoId() + (folder.isEmpty() ? "" : "/" + folder);
        if (safetensors) {
            return new IllegalArgumentException(
                    where
                            + " has no GGUF files (it ships safetensors). jinfer loads GGUF:"
                            + " convert it with llama.cpp's convert_hf_to_gguf.py, or use a GGUF"
                            + " repackage of the same model.");
        }
        return new IllegalArgumentException(
                where + " has no .gguf files. Contains: " + names(files));
    }

    // ---- the cache ----

    /**
     * Where a repository-relative file lives (or would live) in the cache: subfolders preserved, a
     * NAMED revision folded into the repository directory. Every segment is validated because the
     * listing that produced it is remote input.
     */
    static Path pathOf(ModelRef ref, String repoRelative) {
        Path at = repoDir(ref);
        for (String segment : repoRelative.split("/")) {
            if (segment.isEmpty()) {
                continue;
            }
            require(
                    !segment.equals(".")
                            && !segment.equals("..")
                            && segment.indexOf('\\') < 0
                            && segment.indexOf('\0') < 0,
                    "the repository listed a file that would escape the cache: " + repoRelative);
            at = at.resolve(segment);
        }
        return at;
    }

    private static Path repoDir(ModelRef ref) {
        return root().resolve(ref.host().name).resolve(ref.owner()).resolve(ref.cacheRepo());
    }

    /**
     * The cached file this ref selects, or null - checked BEFORE the network, so a warm resolve
     * costs no request, which is also what makes {@code JINFER_OFFLINE} usable.
     *
     * <p>Looks in jinfer's cache first, then READ-ONLY in the HuggingFace hub cache, so a file
     * fetched by {@code hf download}, {@code llama-server -hf} or anything else that writes that
     * layout is found rather than downloaded again. jinfer never writes there: the layout is
     * someone else's, with blobs, symlinks and refs we have no business maintaining.
     */
    private static Path cachedFile(ModelRef ref) throws IOException {
        Path own = cachedIn(ref, folderDir(ref));
        if (own != null) {
            return own;
        }
        Path shared = huggingFaceSnapshot(ref);
        return shared == null ? null : cachedIn(ref, shared);
    }

    /**
     * The snapshot directory this ref names in the HuggingFace hub cache, or null. Honors the same
     * variables every HF client reads, and resolves a branch through {@code refs/} exactly as they
     * do - a ref pinned to a commit needs no indirection at all.
     */
    private static Path huggingFaceSnapshot(ModelRef ref) throws IOException {
        return huggingFaceSnapshot(ref, huggingFaceCache());
    }

    /** Package-visible for its test: the env lookup is the only part a test cannot drive. */
    static Path huggingFaceSnapshot(ModelRef ref, Path hubCache) throws IOException {
        if (ref.host() != ModelRef.Host.HF) {
            return null; // ModelScope's own cache is a different layout; not ours to read
        }
        Path repo = hubCache.resolve("models--" + ref.owner() + "--" + ref.repo()).normalize();
        String revision = ref.revisionOrDefault();
        Path branch = repo.resolve("refs").resolve(revision);
        if (Files.isRegularFile(branch)) {
            revision = Files.readString(branch, StandardCharsets.UTF_8).strip();
        }
        Path snapshot = repo.resolve("snapshots").resolve(revision).resolve(ref.location());
        return Files.isDirectory(snapshot) ? snapshot : null;
    }

    private static Path huggingFaceCache() {
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

    private static Path cachedIn(ModelRef ref, Path dir) throws IOException {
        if (dir == null || !Files.isDirectory(dir)) {
            return null;
        }
        if (!ref.location().isEmpty()) {
            Path named = dir.resolve(nameOf(ref.location()));
            if (Files.isRegularFile(named)) {
                return named; // the ref named a file outright
            }
        }
        try (var entries = Files.list(dir)) {
            List<Path> models =
                    entries.filter(p -> Files.isRegularFile(p))
                            .filter(p -> isModelGguf(p.getFileName().toString()))
                            .sorted()
                            .toList();
            if (models.size() == 1 && ref.quant() == null) {
                return models.get(0);
            }
            List<Path> matches =
                    models.stream()
                            .filter(
                                    p ->
                                            matchesQuant(
                                                    p.getFileName().toString(),
                                                    ref.quantOrDefault()))
                            .toList();
            return matches.size() == 1 ? matches.get(0) : null; // ambiguity goes to the listing
        }
    }

    private static Path fetch(ModelRef ref, RepoFile file) throws IOException {
        if (SPLIT_PART.matcher(file.path()).matches()) {
            throw new UnsupportedOperationException(
                    nameOf(file.path())
                            + " is one part of a split GGUF, which jinfer cannot load. Merge the"
                            + " parts first with llama.cpp's llama-gguf-split --merge, or pick a"
                            + " quant that fits in one file.");
        }
        Path dest = pathOf(ref, file.path());
        requireOnline(ref, dest);
        requireDiskSpace(dest, Fetch.remainingBytes(dest, file.size()));
        tagCacheDirectory(root());
        Fetch.progress()
                .println("download " + ref.host().name + "/" + ref.repoId() + "/" + file.path());
        Fetch.download(
                ref.fileUrl(file.path()), dest, file.size(), file.sha256(), headers(ref.host()));
        return dest;
    }

    private static void requireOnline(ModelRef ref, Path dest) {
        boolean offline =
                "1".equals(System.getenv("JINFER_OFFLINE")) || Boolean.getBoolean("jinfer.offline");
        if (offline) {
            throw new IllegalStateException(
                    ref + " is not cached at " + dest + " and JINFER_OFFLINE forbids downloading");
        }
    }

    /**
     * Refuses a download that cannot fit, BEFORE it starts. Running out of disk partway through a
     * large file is the most reported model-download failure there is (ollama has carried an open
     * report since 2023, huggingface_hub one since 2025), and the error you get when it happens
     * never mentions disk. Ten percent headroom, because a filesystem at 100% is its own kind of
     * broken. {@code JINFER_SKIP_DISK_CHECK=1} opts out - some network mounts report their free
     * space as zero, the escape hatch huggingface_hub had to add for Azure.
     */
    static void requireDiskSpace(Path dest, long size) { // package-visible for its test
        if (size <= 0 || "1".equals(System.getenv("JINFER_SKIP_DISK_CHECK"))) {
            return;
        }
        long free;
        try {
            Path existing = dest.getParent();
            while (existing != null && !Files.exists(existing)) {
                existing = existing.getParent();
            }
            free = existing == null ? -1 : Files.getFileStore(existing).getUsableSpace();
        } catch (IOException unknown) {
            return; // a filesystem that will not say is not one we should refuse over
        }
        long needed = size + size / 10;
        if (free >= 0 && free < needed) {
            throw new IllegalStateException(
                    "not enough space for "
                            + dest.getFileName()
                            + ": it needs "
                            + Fetch.size(size)
                            + " (plus headroom) and "
                            + dest.getRoot()
                            + " has "
                            + Fetch.size(free)
                            + " free. Free some space, or point JINFER_MODELS at another disk.");
        }
    }

    // ---- repository listings ----

    /** The folder a ref searches: its location, unless the location named a file. */
    private static String folderOf(ModelRef ref) throws IOException {
        String location = ref.location();
        if (location.isEmpty()) {
            return "";
        }
        String parent = parentOf(location);
        String last = nameOf(location);
        for (RepoFile file : listFiles(ref, parent)) {
            if (nameOf(file.path()).equals(last)) {
                return parent; // the location is a file: its folder is the parent
            }
        }
        return location;
    }

    /** The same question, answered from the cache's own shape (no network). */
    private static Path folderDir(ModelRef ref) {
        String location = ref.location();
        if (location.isEmpty()) {
            return repoDir(ref);
        }
        Path named = pathOf(ref, location);
        return Files.isRegularFile(named) ? named.getParent() : named;
    }

    private static boolean hasDirectory(ModelRef ref, String parent, String name)
            throws IOException {
        return listDirs(ref, parent).stream().anyMatch(d -> nameOf(d.path()).equals(name));
    }

    private static List<RepoFile> listFiles(ModelRef ref, String path) throws IOException {
        return listing(ref, path).files();
    }

    private static List<RepoDir> listDirs(ModelRef ref, String path) throws IOException {
        return listing(ref, path).dirs();
    }

    private record Listing(List<RepoFile> files, List<RepoDir> dirs) {}

    /** One listing of {@code path} within the repository, at the ref's revision. */
    private static Listing listing(ModelRef ref, String path) throws IOException {
        return switch (ref.host()) {
            case HF -> listHuggingFace(ref, path);
            case MODELSCOPE -> listModelScope(ref, path);
        };
    }

    /**
     * {@code /api/models/owner/repo/tree/<rev>/<path>}: a JSON array, sha256 under {@code lfs.oid}.
     */
    private static Listing listHuggingFace(ModelRef ref, String path) throws IOException {
        String url =
                ref.host().base()
                        + "/api/models/"
                        + ref.repoId()
                        + "/tree/"
                        + ref.revisionOrDefault()
                        + (path.isEmpty() ? "" : "/" + path);
        List<RepoFile> files = new ArrayList<>();
        List<RepoDir> dirs = new ArrayList<>();
        for (Object entry : Json.parseList(get(ref, url))) {
            if (!(entry instanceof Map<?, ?> map)) {
                continue;
            }
            if ("directory".equals(map.get("type"))) {
                dirs.add(new RepoDir(str(map.get("path"))));
                continue;
            }
            if (!"file".equals(map.get("type"))) {
                continue;
            }
            // a plain file's "oid" is a git blob sha1, not content: only LFS entries carry a
            // sha256, and every GGUF worth downloading is one
            String sha256 =
                    map.get("lfs") instanceof Map<?, ?> lfs && lfs.get("oid") instanceof String oid
                            ? oid
                            : null;
            files.add(new RepoFile(str(map.get("path")), num(map.get("size")), sha256));
        }
        return new Listing(files, dirs);
    }

    /** {@code /api/v1/models/owner/repo/repo/files}: {@code Data.Files}, sha256 on every entry. */
    private static Listing listModelScope(ModelRef ref, String path) throws IOException {
        String url =
                ref.host().base()
                        + "/api/v1/models/"
                        + ref.repoId()
                        + "/repo/files?Revision="
                        + ref.revisionOrDefault()
                        + "&Root="
                        + path;
        Map<String, Object> body = Json.parseMap(get(ref, url));
        List<Object> entries =
                Json.queryList(body, "Data", "Files")
                        .orElseThrow(
                                () ->
                                        new IOException(
                                                "unexpected listing for "
                                                        + ref.repoId()
                                                        + " from "
                                                        + url
                                                        + " - API changed?"));
        List<RepoFile> files = new ArrayList<>();
        List<RepoDir> dirs = new ArrayList<>();
        for (Object entry : entries) {
            if (!(entry instanceof Map<?, ?> map)) {
                continue;
            }
            if ("tree".equals(map.get("Type"))) {
                dirs.add(new RepoDir(str(map.get("Path"))));
            } else if ("blob".equals(map.get("Type"))) {
                files.add(
                        new RepoFile(
                                str(map.get("Path")),
                                num(map.get("Size")),
                                map.get("Sha256") instanceof String s && !s.isBlank() ? s : null));
            }
        }
        return new Listing(files, dirs);
    }

    /**
     * A listing GET, translating the failures a user can actually fix: a gated repository, and one
     * that is not there.
     */
    private static String get(ModelRef ref, String url) throws IOException {
        try {
            return Fetch.getString(url, headers(ref.host()));
        } catch (Fetch.HttpStatusException e) {
            if (e.status == 401 || e.status == 403) {
                throw new IllegalArgumentException(
                        ref.repoId()
                                + " is gated or private. Accept its licence at "
                                + ref.repoUrl()
                                + " then set "
                                + ref.host().tokenEnv
                                + (ref.host() == ModelRef.Host.HF
                                        ? " (or log in once with: hf auth login)"
                                        : ""),
                        e);
            }
            if (e.status == 404) {
                throw new IllegalArgumentException(
                        "no repository "
                                + ref.repoId()
                                + " at revision "
                                + ref.revisionOrDefault()
                                + " on "
                                + ref.host().name
                                + " ("
                                + ref.repoUrl()
                                + ")",
                        e);
            }
            throw e;
        }
    }

    /**
     * Credentials for {@code host}: its token variable, and for HuggingFace the token {@code hf
     * auth login} leaves behind - someone who has logged in once should not have to export
     * anything.
     */
    private static Map<String, String> headers(ModelRef.Host host) {
        Map<String, String> headers = new LinkedHashMap<>();
        headers.put("User-Agent", "jinfer-hub");
        String token = System.getenv(host.tokenEnv);
        if ((token == null || token.isBlank()) && host == ModelRef.Host.HF) {
            token = huggingFaceTokenFile();
        }
        if (token != null && !token.isBlank()) {
            headers.put("Authorization", "Bearer " + token.strip());
        }
        return headers;
    }

    /** {@code $HF_TOKEN_PATH} &gt; {@code $HF_HOME/token} &gt; the default, as every HF client. */
    private static String huggingFaceTokenFile() {
        String explicit = System.getenv("HF_TOKEN_PATH");
        String home = System.getenv("HF_HOME");
        Path token;
        if (explicit != null && !explicit.isBlank()) {
            token = Path.of(explicit);
        } else if (home != null && !home.isBlank()) {
            token = Path.of(home, "token");
        } else {
            token = Path.of(System.getProperty("user.home"), ".cache", "huggingface", "token");
        }
        try {
            return Files.isRegularFile(token)
                    ? Files.readString(token, StandardCharsets.UTF_8)
                    : null;
        } catch (IOException e) {
            return null; // an unreadable token file is not a reason to fail a public download
        }
    }

    // ---- matching ----

    /**
     * Filename prefixes that mark a file as a COMPANION of a model rather than a model: a media
     * projector, a draft head. This is a naming convention of GGUF repositories - the same kind of
     * knowledge as a quant name - and it exists so that a quant search cannot answer with a
     * projector, which is a real collision in repositories shipping both {@code x-f16.gguf} and
     * {@code mmproj-model-f16.gguf}.
     */
    private static final List<String> COMPANIONS = List.of("mmproj", "mtp");

    /** A model file: a .gguf that is not one of a model's companions. */
    private static boolean isModelGguf(String fileName) {
        return fileName.toLowerCase(Locale.ROOT).endsWith(".gguf")
                && COMPANIONS.stream().noneMatch(prefix -> startsWith(fileName, prefix));
    }

    private static boolean startsWith(String fileName, String prefix) {
        return fileName.toLowerCase(Locale.ROOT).startsWith(prefix.toLowerCase(Locale.ROOT));
    }

    /**
     * Whether {@code fileName} carries {@code quant} between filename delimiters, so {@code Q4_K_M}
     * cannot answer for {@code Q4_K_XL}. A PREFIX still matches ({@code Q4_K} finds {@code
     * Q4_K_XL}), which is deliberate: it either identifies one file or the caller is shown every
     * candidate. Nothing here picks silently between two quants of one model, because that is how a
     * benchmark ends up measuring a file nobody chose.
     */
    static boolean matchesQuant(String fileName, String quant) { // package-visible for its test
        String name = fileName.toLowerCase(Locale.ROOT);
        String needle = quant.toLowerCase(Locale.ROOT);
        for (int at = 0; (at = name.indexOf(needle, at)) >= 0; at++) {
            boolean left = at == 0 || isSeparator(name.charAt(at - 1));
            int end = at + needle.length();
            boolean right = end == name.length() || isSeparator(name.charAt(end));
            if (left && right) {
                return true;
            }
        }
        return false;
    }

    private static boolean isSeparator(char c) {
        return c == '-' || c == '.' || c == '_';
    }

    // ---- small helpers ----

    private static String nameOf(String path) {
        int slash = path.lastIndexOf('/');
        return slash < 0 ? path : path.substring(slash + 1);
    }

    private static String parentOf(String path) {
        int slash = path.lastIndexOf('/');
        return slash < 0 ? "" : path.substring(0, slash);
    }

    /**
     * The candidates as copy-pasteable refs with their sizes, smallest first - a menu rather than a
     * list dumped into a sentence. Ambiguity is always resolved by NAMING one, never by guessing,
     * so the message's job is to make naming a copy and a paste.
     */
    private static String menu(ModelRef ref, List<RepoFile> matches) {
        List<RepoFile> sorted =
                matches.stream()
                        .sorted(
                                Comparator.comparingLong(RepoFile::size)
                                        .thenComparing(RepoFile::path))
                        .toList();
        int width = sorted.stream().mapToInt(f -> f.path().length()).max().orElse(0);
        StringBuilder out = new StringBuilder();
        for (RepoFile f : sorted) {
            out.append("\n  ")
                    .append(ref.host().name)
                    .append('/')
                    .append(ref.repoId())
                    .append('/')
                    .append(f.path())
                    .append(" ".repeat(width - f.path().length()))
                    .append("   ")
                    .append(Fetch.size(f.size()));
        }
        return out.toString();
    }

    private static List<String> names(List<RepoFile> files) {
        return files.stream().map(f -> nameOf(f.path())).sorted().toList();
    }

    private static String str(Object value) {
        return value instanceof String s ? s : "";
    }

    private static long num(Object value) {
        return value instanceof Number n ? n.longValue() : -1;
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
