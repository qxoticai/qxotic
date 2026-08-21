package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.System.Logger.Level;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
 * cache. Flat and obvious on purpose: {@code ls} and {@code rm -rf} are the management commands.
 *
 * <p>A store is an INSTANCE: {@link #standard()} for the ambient root ({@code -Djinfer.models} &gt;
 * {@code $JINFER_MODELS} &gt; the platform's cache directory) with the two shipped sources, {@link
 * #of} to build your own - a scratch root for a test, or a store with NO sources, which is the
 * offline story made explicit: cache hits resolve, misses fail without touching the network. {@code
 * standard()} builds fresh on every call, so a test that sets {@code jinfer.models} sees it.
 *
 * <p>SOURCES are tried in the order they were given; the first that serves the ref wins. A source
 * that fails (network down, repository answered "no") is not silently skipped: the fallback is
 * logged at WARNING and the success of a non-primary source at INFO. A miss everywhere throws the
 * LAST source's own answer - the most specific message there is, and the one a single-source store
 * would have given - with every earlier failure loud in the log.
 *
 * <p>ONE exception to the flat layout, for sharing: with the DEFAULT root (by value - {@code
 * of(standard().root())} keeps it), an {@code hf.co} download is written into the HuggingFace hub
 * cache in its own layout, so the bytes are immediately visible to llama.cpp, {@code hf download}
 * and everything else that reads that layout - and theirs to us, which the read side always does.
 * An EXPLICIT root opts out: it says "my cache lives here, all of it", and it is also the
 * documented escape hatch for a full disk, so nothing may leak elsewhere. ModelScope and plain URLs
 * always use the flat layout - there is no shared convention to join.
 *
 * <p>Format policy lives HERE, not in the grammar. {@link ModelRef} parses any repository path;
 * this class knows that jinfer loads GGUF, and refuses a repository that ships none BEFORE any
 * bytes move rather than after a caller waits for twenty gigabytes.
 *
 * <p>READ-ONLY is a supported deployment: point {@code root} at a mounted, pre-populated cache.
 * {@link #find}, {@link #cached} and cache hits through {@link #resolve} all work; a miss fails
 * with "cache root is not writable"; {@link #evict} reports the filesystem's refusal. Sources are
 * read-only too - no source ever writes into the store beyond the one path it was handed.
 *
 * <p>Nothing in the inference engine calls this. Resolution happens in a CLI, before {@code
 * Models.load}, so a Java caller that passes a path gets exactly that path and no library load ever
 * touches the network.
 */
public final class ModelStore {

    private static final System.Logger LOG = System.getLogger(ModelStore.class.getName());

    /** llama.cpp's default, so a bare repository means the same file in both tools. */
    private static final String DEFAULT_QUANT = "Q4_K_M";

    private final Path root;
    private final boolean hubShare;
    private final List<ModelSource> sources;

    private ModelStore(Path root, boolean hubShare, List<ModelSource> sources) {
        this.root = root;
        this.hubShare = hubShare;
        this.sources = sources;
    }

    /**
     * The ambient store: the root from {@code -Djinfer.models} / {@code $JINFER_MODELS} / the
     * platform cache directory, with HuggingFace and ModelScope as sources. Built FRESH on every
     * call - the ambient lookups (property, env) happen now, so a test that sets {@code
     * jinfer.models} gets a store that honors it.
     */
    public static ModelStore standard() {
        return of(ambientRoot(), new HuggingFaceSource(), new ModelScopeSource());
    }

    /**
     * A store over {@code root} fed by {@code sources}, in the order given. No sources means a
     * store that can only serve what the root already holds - the offline deployment, without a
     * flag. The hub-cache write-through applies when {@code root} IS the platform default by value,
     * wherever the caller got it.
     */
    public static ModelStore of(Path root, ModelSource... sources) {
        Objects.requireNonNull(root, "root");
        Path normalized = root.toAbsolutePath().normalize();
        boolean hubShare =
                normalized.equals(platformCache().resolve("jinfer").toAbsolutePath().normalize());
        return new ModelStore(normalized, hubShare, List.of(sources));
    }

    /** Where downloads land, and the first place a ref is looked for. */
    public Path root() {
        return root;
    }

    static Path ambientRoot() {
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

    /** Whether {@code model} is a model ref ({@code host/owner/repo[@rev][/path][:quant]}). */
    public static boolean isRef(String model) {
        return ModelRef.isRef(model);
    }

    /**
     * The one door for a remote model. A model ref names its host, and nothing else qualifies: a
     * local path, a URL, and a bare {@code owner/repo} each get their own remedy.
     */
    public static void requireRef(String model) {
        if (isRef(model)) {
            return;
        }
        String value = model == null ? "" : model.strip();
        if (value.isEmpty()) {
            throw new IllegalArgumentException(
                    "a model ref is required: hf.co/owner/repo[:quant] or"
                            + " modelscope.cn/owner/repo[:quant]");
        }
        if (value.contains("://")) {
            throw new IllegalArgumentException(
                    "'"
                            + model
                            + "' is a URL, not a model ref. Download it first, then pass the file"
                            + " with modelPath(...).");
        }
        if (isLocalPathShape(value)) {
            throw new IllegalArgumentException(
                    "'"
                            + model
                            + "' is a local path, not a model ref. Use modelPath(...) for a local"
                            + " file, or name a host: hf.co/owner/repo[:quant].");
        }
        if (value.indexOf('/') > 0 && value.indexOf('/') == value.lastIndexOf('/')) {
            throw new IllegalArgumentException(
                    "'" + model + "' is missing its host. Did you mean hf.co/" + model + "?");
        }
        throw new IllegalArgumentException(
                "'"
                        + model
                        + "' is not a model ref. A model ref names its host, for example"
                        + " hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M or"
                        + " modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0");
    }

    private static String quantOf(ModelRef ref) {
        return ref.quant() == null ? DEFAULT_QUANT : ref.quant();
    }

    private static boolean isLocalPathShape(String value) {
        return value.startsWith("/")
                || value.startsWith(".")
                || value.startsWith("~")
                || value.indexOf('\\') >= 0
                || (value.length() >= 2
                        && Character.isLetter(value.charAt(0))
                        && value.charAt(1) == ':');
    }

    /** A plain {@code http(s)} URL, distinct from a model ref. */
    private static boolean isHttpUrl(String value) {
        if (value == null) {
            return false;
        }
        String scheme = value.indexOf("://") < 0 ? null : value.substring(0, value.indexOf("://"));
        return "http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme);
    }

    /**
     * A local path or a remote ref, told apart by ONE visible rule: a ref names its host,
     * everything else is a file on this machine.
     */
    public Path resolve(String pathOrRef) {
        if (ModelRef.isRef(pathOrRef)) {
            return resolveRef(ModelRef.parse(pathOrRef)); // a repository a source may talk to
        }
        if (isHttpUrl(pathOrRef)) {
            return url(pathOrRef); // any other URL: bytes, and nothing else
        }
        Path local = localFile(pathOrRef);
        if (local == null) {
            throw new IllegalArgumentException(
                    "no such model file: '"
                            + pathOrRef
                            + "'. A model ref names its host, for example"
                            + " hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M or"
                            + " modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0");
        }
        return local;
    }

    /**
     * {@link #resolve} for several arguments at once, downloading the missing ones CONCURRENTLY - a
     * cold start with a model plus an mmproj pays the slower download, not the sum. Paths return in
     * input order; the first failure cancels the rest (their partial state resumes later) and is
     * the one thrown. Warm entries never spawn anything: one argument short-circuits entirely.
     */
    // ponytail: no dedup, no aggregate progress bar (concurrent downloads print named lines, see
    // Fetch.Progress), no shared disk-space budget. Add each when someone actually hits it.
    public List<Path> resolveAll(List<String> pathOrRefs) {
        if (pathOrRefs.size() <= 1 || Fetch.oneAtATime()) {
            return pathOrRefs.stream().map(this::resolve).toList();
        }
        // ponytail: at most 4 files in flight (x up to 8 chunk connections each, see
        // JINFER_DOWNLOAD_THREADS). A fixed constant, not a knob - add the env var when a real
        // pull is throttled by it.
        try (var pool = Executors.newFixedThreadPool(Math.min(4, pathOrRefs.size()))) {
            List<Future<Path>> futures =
                    pathOrRefs.stream().map(r -> pool.submit(() -> resolve(r))).toList();
            List<Path> paths = new ArrayList<>(futures.size());
            try {
                for (var future : futures) {
                    paths.add(future.get());
                }
            } catch (ExecutionException e) {
                futures.forEach(f -> f.cancel(true)); // siblings leave resumable .part files
                switch (e.getCause()) {
                    case RuntimeException runtime -> throw runtime;
                    case Error error -> throw error;
                    case Throwable other -> throw new IllegalStateException(other);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                futures.forEach(f -> f.cancel(true));
                throw new IllegalStateException("interrupted while resolving models", e);
            }
            return paths;
        }
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
    private Path url(String url) {
        URI uri;
        try {
            uri = URI.create(url);
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

        Path dest = root.resolve(uri.getHost());
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
        // before the SIZE probe, not just the download: offline means no request at all
        requireOnlineFor(url, dest);
        Map<String, String> headers = Map.of("User-Agent", "jinfer-hub");
        long size = Fetch.sizeOf(url, headers);
        requireWritable(dest);
        requireDiskSpace(dest, Fetch.remainingBytes(dest, size));
        tagCacheDirectory(root);
        Fetch.announce(
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

    /**
     * {@link #resolve} without the download: the cached file this ref or path names, or empty. Same
     * lookup order as resolve - jinfer's own cache first, then the HuggingFace hub cache - so a hit
     * here is exactly the file resolve would return without fetching. Never touches the network,
     * which is what makes it safe to call from a test or a {@code list} command. Plain URLs are
     * refused: they carry no checksum, so finding one in the cache would vouch for bytes nothing
     * verified.
     */
    public Optional<Path> find(String pathOrRef) {
        if (ModelRef.isRef(pathOrRef)) {
            try {
                return Optional.ofNullable(cachedFile(ModelRef.parse(pathOrRef)));
            } catch (IOException e) {
                throw new UncheckedIOException("could not look up " + pathOrRef + ": " + e, e);
            }
        }
        if (isHttpUrl(pathOrRef)) {
            throw new IllegalArgumentException(
                    "plain URLs can only be resolved (they carry no checksum): " + pathOrRef);
        }
        return Optional.ofNullable(localFile(pathOrRef));
    }

    /**
     * The file this string names, or null when it names none. A directory is refused by name rather
     * than reported as missing.
     */
    private static Path localFile(String path) {
        try {
            Path candidate = Path.of(path);
            if (Files.isDirectory(candidate)) {
                throw new IllegalArgumentException(
                        "'" + path + "' is a directory; a model or companion is a single file");
            }
            return Files.isRegularFile(candidate) ? candidate : null;
        } catch (InvalidPathException notAPath) {
            return null;
        }
    }

    private static void requireOnlineFor(String what, Path dest) {
        if (offline()) {
            throw new IllegalStateException(
                    what + " is not cached at " + dest + " and JINFER_OFFLINE forbids downloading");
        }
    }

    private static boolean offline() {
        return "1".equals(System.getenv("JINFER_OFFLINE")) || Boolean.getBoolean("jinfer.offline");
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
     *
     * <p>GGUF files in the shared HuggingFace hub cache are listed too, as refs - they resolve
     * without a download, whichever tool put them there, and with the default root that tool may
     * well have been jinfer itself. Only {@code .gguf} files: that cache also holds every tokenizer
     * and config the Python stack ever fetched, none of which is a {@code --model}.
     */
    public List<Cached> cached() {
        List<Cached> all = new ArrayList<>(ownCached());
        all.addAll(Hub.cached(Hub.cache()));
        return all.stream().distinct().sorted(Comparator.comparing(Cached::ref)).toList();
    }

    /** One cache entry: the ref (or path) to ask for it again, and its size on disk. */
    public record Cached(String ref, long sizeBytes) {}

    private List<Cached> ownCached() {
        if (!Files.isDirectory(root)) {
            return List.of();
        }
        try (var walk = Files.walk(root)) {
            return walk.filter(Files::isRegularFile)
                    .filter(file -> isCachedModel(root.relativize(file)))
                    .map(file -> new Cached(asRefOrPath(root, root.relativize(file)), sizeOf(file)))
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "could not read the model cache at " + root + ": " + e, e);
        }
    }

    static long sizeOf(Path file) {
        try {
            return Files.size(file);
        } catch (IOException unreadable) {
            return 0; // a size the filesystem will not state is not worth failing a listing over
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
    public boolean evict(String pathOrRef) {
        if (!ModelRef.isRef(pathOrRef)) {
            return false; // a file you passed by path is yours; this cache never deletes it
        }
        ModelRef ref = ModelRef.parse(pathOrRef);
        try {
            Path cached = cachedFile(ref);
            if (cached == null) {
                return false;
            }
            if (cached.toAbsolutePath().startsWith(root)) {
                return Files.deleteIfExists(cached); // our flat cache: the file is the entry
            }
            // the shared hub cache: jinfer writes there too now, so it may also remove there -
            // but only THIS entry, and its blob only once nothing else in the repo links it.
            // Everything beyond that belongs to other snapshots and other tools.
            Path hub = Hub.cache().toAbsolutePath().normalize();
            if (ref.host().equals(ModelRef.Host.HF.name)
                    && cached.toAbsolutePath().normalize().startsWith(hub)) {
                return Hub.evict(cached);
            }
            return false;
        } catch (IOException e) {
            throw new UncheckedIOException("could not evict " + ref + ": " + e, e);
        }
    }

    // ---- resolution ----

    /** The cached file for {@code ref}, fetching it from the first source that can. */
    private Path resolveRef(ModelRef ref) {
        try {
            Path cached = cachedFile(ref);
            if (cached != null) {
                return cached;
            }
        } catch (IOException e) {
            // the cause carries the only actionable part (refused proxy, DNS, TLS, timeout)
            throw new UncheckedIOException("could not resolve " + ref + ": " + e, e);
        }
        // before the LISTING, not just the download: offline means no request at all
        requireOnline(ref, folderDir(ref));
        List<ModelSource> serving = sources.stream().filter(s -> s.supports(ref)).toList();
        if (serving.isEmpty()) {
            throw new UncheckedIOException(
                    new IOException(
                            ref
                                    + " is not cached, and no source in this store serves "
                                    + ref.host()
                                    + (sources.isEmpty()
                                            ? " (this store was built with no sources -"
                                                    + " ModelStore.of(root) is the offline store)"
                                            : "")));
        }
        for (int i = 0; i < serving.size(); i++) {
            ModelSource source = serving.get(i);
            boolean last = i == serving.size() - 1;
            try {
                Path path = fetchPipeline(ref, source);
                if (i > 0) {
                    LOG.log(Level.INFO, "{0} resolved by {1}", ref, source);
                }
                return path;
            } catch (IllegalArgumentException e) {
                if (last || !(e.getCause() instanceof Fetch.HttpStatusException)) {
                    throw e; // the best answer there is: the ref's fault, or the last word
                }
                warn(source, ref, e);
            } catch (IOException e) {
                if (last) {
                    throw new UncheckedIOException("could not resolve " + ref + ": " + e, e);
                }
                warn(source, ref, e);
            }
        }
        throw new AssertionError("unreachable: the last source either serves or throws");
    }

    /** A fallback is never silent: who failed, on what, and why, before the next source tries. */
    private static void warn(ModelSource source, ModelRef ref, Exception e) {
        LOG.log(Level.WARNING, "{0} could not serve {1}: {2}", source, ref, e.getMessage());
    }

    private Path fetchPipeline(ModelRef ref, ModelSource source) throws IOException {
        RemoteFile file = select(ref, source);
        if (SPLIT_PART.matcher(file.path()).matches()) {
            throw new UnsupportedOperationException(
                    nameOf(file.path())
                            + " is one part of a split GGUF, which jinfer cannot load. Merge the"
                            + " parts first with llama.cpp's llama-gguf-split --merge, or pick a"
                            + " quant that fits in one file.");
        }
        if (hubShare && ref.host().equals(ModelRef.Host.HF.name) && Hub.isSha256(file.sha256())) {
            String commit = Hub.commit(ref);
            if (commit != null) {
                return Hub.fetchInto(ref, file, commit, Hub.cache());
            }
            // no commit means no snapshot directory to link under; the flat layout still works
        }
        Path dest = pathOf(ref, file.path());
        requireWritable(dest);
        requireDiskSpace(dest, Fetch.remainingBytes(dest, file.sizeBytes()));
        tagCacheDirectory(root);
        Fetch.announce("download " + ref.host() + "/" + ref.repoId() + "/" + file.path());
        source.fetch(ref, file, dest);
        return dest;
    }

    /**
     * A store rooted on a read-only mount still serves every cache hit; a MISS is where the write
     * would happen, and that is refused by name rather than as the staging error the source would
     * surface.
     */
    private void requireWritable(Path dest) {
        Path dir = dest.getParent();
        while (dir != null && !Files.exists(dir)) {
            dir = dir.getParent();
        }
        if (dir == null || !Files.isWritable(dir)) {
            throw new UncheckedIOException(new IOException("cache root is not writable: " + root));
        }
    }

    /**
     * The one file {@code ref} selects, from the source's listing.
     *
     * <p>The listing - never a file extension - decides whether the ref's path names a file or a
     * folder, which is what keeps the grammar free of formats: {@code .safetensors} and a folder
     * called {@code Qwen2.5} both work without a special case. The parent is listed first; if the
     * path's last segment is not a FILE there, the path is treated as a folder and listed - a
     * failure or an empty answer means the path names nothing, and the parent's contents make the
     * menu.
     */
    RemoteFile select(ModelRef ref, ModelSource source) throws IOException {
        String location = ref.path();
        if (!location.isEmpty()) {
            String parent = parentOf(location);
            String last = nameOf(location);
            List<RemoteFile> siblings = source.list(ref, parent);
            for (RemoteFile file : siblings) {
                if (nameOf(file.path()).equals(last)) {
                    require(
                            ref.quant() == null,
                            location
                                    + " already names a file, so ':"
                                    + ref.quant()
                                    + "' has nothing to choose - drop one of them");
                    return file;
                }
            }
            String nothingThere =
                    "no '"
                            + last
                            + "' in "
                            + ref.repoId()
                            + (parent.isEmpty() ? "" : "/" + parent)
                            + ". Contains: "
                            + names(siblings);
            List<RemoteFile> inside;
            try {
                inside = source.list(ref, location);
            } catch (IOException | IllegalArgumentException notAFolder) {
                throw new IllegalArgumentException(nothingThere);
            }
            if (inside.isEmpty()) {
                throw new IllegalArgumentException(nothingThere);
            }
            return byQuant(ref, location, inside);
        }
        return byQuant(ref, "", source.list(ref, ""));
    }

    /**
     * The single file matching this ref's quant inside {@code folder}, or a message saying why not.
     */
    private RemoteFile byQuant(ModelRef ref, String folder, List<RemoteFile> files) {
        List<RemoteFile> models =
                files.stream().filter(f -> isModelGguf(nameOf(f.path()))).toList();
        if (models.isEmpty()) {
            throw noGguf(ref, folder, files);
        }
        // one GGUF and no quant asked for: that is the model, whatever it is called. An EXPLICIT
        // quant never falls back - a caller who named one and got another has been lied to.
        if (models.size() == 1 && ref.quant() == null) {
            return models.get(0);
        }
        List<RemoteFile> matches =
                models.stream().filter(f -> matchesQuant(nameOf(f.path()), quantOf(ref))).toList();
        if (matches.size() == 1) {
            return matches.get(0);
        }
        if (matches.isEmpty()) {
            throw new IllegalArgumentException(
                    "no "
                            + quantOf(ref)
                            + " in "
                            + ref.repoId()
                            + (folder.isEmpty() ? "" : "/" + folder)
                            + ". Available: "
                            + names(models));
        }
        throw new IllegalArgumentException(
                quantOf(ref)
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
            ModelRef ref, String folder, List<RemoteFile> files) {
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

    private static final Pattern SPLIT_PART =
            Pattern.compile(".*-\\d{5}-of-\\d{5}\\.gguf$", Pattern.CASE_INSENSITIVE);

    /**
     * Where a repository-relative file lives (or would live) in the cache: subfolders preserved, a
     * NAMED revision folded into the repository directory. Every segment is validated because the
     * listing that produced it is remote input.
     */
    Path pathOf(ModelRef ref, String repoRelative) {
        return under(repoDir(ref), repoRelative);
    }

    /** {@code repoRelative} resolved beneath {@code base}, every segment checked for escape. */
    static Path under(Path base, String repoRelative) {
        Path at = base;
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

    private Path repoDir(ModelRef ref) {
        return root.resolve(ref.host()).resolve(ref.owner()).resolve(ref.cacheRepo());
    }

    /**
     * The cached file this ref selects, or null - checked BEFORE the network, so a warm resolve
     * costs no request, which is also what makes {@code JINFER_OFFLINE} usable.
     *
     * <p>Looks in jinfer's own cache first, then in the HuggingFace hub cache, so a file fetched by
     * {@code hf download}, {@code llama-server -hf} or anything else that writes that layout is
     * found rather than downloaded again - and with the default root, jinfer's own downloads land
     * there too ({@link Hub#fetchInto}).
     */
    private Path cachedFile(ModelRef ref) throws IOException {
        Path own = cachedIn(ref, folderDir(ref));
        if (own != null) {
            return own;
        }
        Path shared = Hub.snapshot(ref);
        return shared == null ? null : cachedIn(ref, shared);
    }

    private static Path cachedIn(ModelRef ref, Path dir) throws IOException {
        if (dir == null || !Files.isDirectory(dir)) {
            return null;
        }
        if (!ref.path().isEmpty()) {
            Path named = dir.resolve(nameOf(ref.path()));
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
                            .filter(p -> matchesQuant(p.getFileName().toString(), quantOf(ref)))
                            .toList();
            return matches.size() == 1 ? matches.get(0) : null; // ambiguity goes to the listing
        }
    }

    private static void requireOnline(ModelRef ref, Path dest) {
        if (offline()) {
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

    /** The folder a ref would search on disk: its path, unless the path named a file. */
    private Path folderDir(ModelRef ref) {
        String location = ref.path();
        if (location.isEmpty()) {
            return repoDir(ref);
        }
        Path named = pathOf(ref, location);
        return Files.isRegularFile(named) ? named.getParent() : named;
    }

    // ---- matching ----

    /**
     * Filename prefixes that mark a file as a COMPANION of a model rather than a model: a media
     * projector, a draft head. This is a naming convention of GGUF repositories - the same kind of
     * knowledge as a quant name - and it exists so that a quant search cannot answer with a
     * projector, which is a real collision in repositories shipping both {@code model-f16.gguf} and
     * {@code mmproj-model-f16.gguf}.
     */
    private static final List<String> COMPANIONS = List.of("mmproj", "mtp");

    static boolean isGguf(String fileName) {
        return fileName.toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    /** A model file: a .gguf that is not one of a model's companions. */
    private static boolean isModelGguf(String fileName) {
        return isGguf(fileName)
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

    static String nameOf(String path) {
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
    private static String menu(ModelRef ref, List<RemoteFile> matches) {
        List<RemoteFile> sorted =
                matches.stream()
                        .sorted(
                                Comparator.comparingLong(RemoteFile::sizeBytes)
                                        .thenComparing(RemoteFile::path))
                        .toList();
        int width = sorted.stream().mapToInt(f -> f.path().length()).max().orElse(0);
        StringBuilder out = new StringBuilder();
        for (RemoteFile f : sorted) {
            out.append("\n  ")
                    .append(ref.host())
                    .append('/')
                    .append(ref.repoId())
                    .append('/')
                    .append(f.path())
                    .append(" ".repeat(width - f.path().length()))
                    .append("   ")
                    .append(Fetch.size(f.sizeBytes()));
        }
        return out.toString();
    }

    private static List<String> names(List<RemoteFile> files) {
        return files.stream().map(f -> nameOf(f.path())).sorted().toList();
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
