package com.qxotic.jinfer.hub;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * A model reference: a URL with the scheme left off.
 *
 * <pre>
 *   [https://] host / owner/repo [@revision] [/path] [:quant]
 *
 *   hf.co/unsloth/gemma-4-E2B-it-GGUF                      Q4_K_M at the repository root
 *   hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0                 that quant
 *   hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf      that exact file
 *   hf.co/ggml-org/models/bert-bge-small:F16               that quant inside a subfolder
 *   hf.co/ggml-org/models@a1b2c3d/bert-bge-small           at a pinned revision
 *   modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0                the other host
 * </pre>
 *
 * <p>Position carries the meaning, so nothing is polymorphic: the path says WHERE, the {@code
 * :quant} tag says WHICH, and neither can stand in for the other. A ref names its host, so remote
 * and local are told apart by a closed table rather than by probing the filesystem - the same
 * argument therefore means the same thing on every machine, in a script and in a config file.
 *
 * <p>Refs are also paste-tolerant: a browser URL, with its scheme, its {@code huggingface.co}
 * spelling, its {@code /tree/}, {@code /blob/} or {@code /resolve/} view and its {@code
 * ?download=true}, normalizes to the same ref as the short form, and therefore to the same cache
 * entry.
 *
 * <p>NOTHING here knows about file formats. Whether a path names a file or a folder is answered by
 * the repository listing, not by an extension, so {@code .gguf}, {@code .safetensors} and anything
 * else parse identically. What jinfer is willing to LOAD is a separate question, answered once in
 * {@link ModelStore}.
 */
record ModelRef(
        Host host, String owner, String repo, String revision, String location, String quant) {

    /**
     * A host jinfer can talk to as a repository. Both serve a repository the same way - {@code
     * <host><prefix>/owner/repo/resolve/<revision>/<file>}, redirecting to a signed CDN URL - so
     * only the listing API and these few fields differ.
     */
    enum Host {
        HF(
                "hf.co",
                "https://huggingface.co",
                "",
                "main",
                "HF_TOKEN",
                "HF_ENDPOINT",
                "huggingface.co"),
        MODELSCOPE(
                "modelscope.cn",
                "https://modelscope.cn",
                "/models",
                "master",
                "MODELSCOPE_API_TOKEN",
                "MODELSCOPE_ENDPOINT");

        /** The canonical name: what a ref says, and the cache's first directory. */
        final String name;

        /** Other spellings that mean this host - what a browser's address bar may say instead. */
        final List<String> aliases;

        final String prefix;

        /** The branch a ref means when it names none - the two hosts genuinely disagree. */
        final String defaultRevision;

        final String tokenEnv;
        private final String defaultBase;
        private final String endpointEnv;

        Host(
                String name,
                String defaultBase,
                String prefix,
                String defaultRevision,
                String tokenEnv,
                String endpointEnv,
                String... aliases) {
            this.name = name;
            this.aliases = List.of(aliases);
            this.defaultBase = defaultBase;
            this.prefix = prefix;
            this.defaultRevision = defaultRevision;
            this.tokenEnv = tokenEnv;
            this.endpointEnv = endpointEnv;
        }

        /**
         * The base URL, honoring this host's mirror variable ({@code HF_ENDPOINT} is what every
         * HuggingFace client reads, and how someone behind a blocked or slow route reaches a
         * mirror). The cache directory stays {@link #name} whatever the mirror is: a mirror serves
         * the same repository, so its files belong in the same place.
         */
        String base() {
            String endpoint = System.getenv(endpointEnv);
            if (endpoint == null || endpoint.isBlank()) {
                return defaultBase;
            }
            String trimmed = endpoint.strip();
            return trimmed.endsWith("/") ? trimmed.substring(0, trimmed.length() - 1) : trimmed;
        }
    }

    /**
     * The host {@code name} spells, or null. Each row declares its own spellings, so adding a host
     * is one enum constant; a leading {@code www.} is stripped by rule rather than enumerated,
     * covering every host's browser spelling at once. (The listing parser lives in {@code
     * ModelStore}, where an exhaustive switch makes a row without one a compile error - this class
     * stays pure grammar.)
     */
    private static Host host(String name) {
        String canonical = name.toLowerCase(Locale.ROOT);
        if (canonical.startsWith("www.")) {
            canonical = canonical.substring("www.".length());
        }
        for (Host host : Host.values()) {
            if (host.name.equals(canonical) || host.aliases.contains(canonical)) {
                return host;
            }
        }
        return null;
    }

    /** Whether this is a host a ref may name - the first segment of every repository ref. */
    static boolean isKnownHost(String name) {
        return host(name) != null;
    }

    /** The canonical host names, for a message that has to list what jinfer knows. */
    static String knownHosts() {
        return java.util.Arrays.stream(Host.values())
                .map(h -> h.name)
                .collect(java.util.stream.Collectors.joining(", "));
    }

    /** The host part of an explicit {@code scheme://host/...}, or null when there is no scheme. */
    static String hostOfUrl(String candidate) {
        int scheme = candidate.indexOf("://");
        if (scheme <= 0) {
            return null;
        }
        String rest = candidate.substring(scheme + 3);
        int slash = rest.indexOf('/');
        return slash > 0 ? rest.substring(0, slash) : rest;
    }

    /** llama.cpp's default, so a bare repository means the same file in both tools. */
    static final String DEFAULT_QUANT = "Q4_K_M";

    /** The Hub view segments that carry a revision, so a pasted URL from any tab works. */
    private static final List<String> VIEWS = List.of("blob", "resolve", "tree", "raw");

    ModelRef {
        component(owner, "owner");
        component(repo, "repository");
        if (revision != null) {
            component(revision, "revision");
        }
        if (quant != null) {
            component(quant, "quant");
        }
        for (String segment : segments(location)) {
            component(segment, "path segment");
        }
    }

    /**
     * Whether {@code candidate} names a remote model: its first segment is a known host. THE one
     * rule separating remote from local, and it consults a table rather than the filesystem.
     *
     * <p>A Windows drive letter is one character and no host is, so {@code C:\models\x.gguf} can
     * never be read as a ref.
     */
    static boolean isRef(String candidate) {
        if (candidate == null || candidate.isBlank()) {
            return false;
        }
        String rest = withoutScheme(candidate.strip());
        int slash = rest.indexOf('/');
        return slash > 0 && host(rest.substring(0, slash)) != null;
    }

    /** Parses a ref, per the grammar in the class note. */
    static ModelRef parse(String ref) {
        require(ref != null && !ref.isBlank(), "empty model ref");
        String rest = withoutScheme(stripQuery(ref.strip()));
        int slash = rest.indexOf('/');
        Host host = slash > 0 ? host(rest.substring(0, slash)) : null;
        require(host != null, shape(ref));
        rest = rest.substring(slash + 1);
        while (rest.endsWith("/")) {
            rest = rest.substring(0, rest.length() - 1);
        }
        List<String> parts = segments(rest);
        require(parts.size() >= 2, shape(ref));

        // the quant is a colon in the FINAL segment; a URL port lives in the host, already consumed
        String last = parts.get(parts.size() - 1);
        String quant = null;
        int colon = last.lastIndexOf(':');
        if (colon >= 0) {
            quant = last.substring(colon + 1);
            parts.set(parts.size() - 1, last.substring(0, colon));
            require(!parts.get(parts.size() - 1).isEmpty(), shape(ref));
        }

        String owner = parts.get(0);
        String repo = parts.get(1);
        String revision = null;
        List<String> path;
        if (parts.size() >= 4 && VIEWS.contains(parts.get(2))) {
            // a pasted Hub URL: .../<view>/<revision>/<path...>
            revision = parts.get(3);
            path = parts.subList(4, parts.size());
        } else {
            int at = repo.indexOf('@');
            if (at >= 0) {
                revision = repo.substring(at + 1);
                repo = repo.substring(0, at);
            }
            path = parts.subList(2, parts.size());
        }
        return new ModelRef(host, owner, repo, revision, String.join("/", path), quant);
    }

    private static String withoutScheme(String ref) {
        int scheme = ref.indexOf("://");
        return scheme < 0 ? ref : ref.substring(scheme + 3);
    }

    /**
     * Drops a query and fragment. The Hub's download button appends {@code ?download=true}, and no
     * part of a repository ref is ever carried in a query.
     */
    private static String stripQuery(String ref) {
        int cut = ref.length();
        for (int i = 0; i < ref.length(); i++) {
            if (ref.charAt(i) == '?' || ref.charAt(i) == '#') {
                cut = i;
                break;
            }
        }
        return ref.substring(0, cut);
    }

    private static List<String> segments(String path) {
        List<String> parts = new ArrayList<>();
        for (String segment : path.split("/")) {
            if (!segment.isEmpty()) {
                parts.add(segment);
            }
        }
        return parts;
    }

    /** What a ref looks like - the message every malformed one gets, so it teaches the grammar. */
    private static String shape(String ref) {
        return "not a model ref: '"
                + ref
                + "'. A REMOTE model names its host: hf.co/owner/repo[:quant] or"
                + " modelscope.cn/owner/repo[:quant], for example"
                + " hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M. Anything else is a local file path.";
    }

    /**
     * Every field becomes a directory or file name under the cache root, so every field must be ONE
     * path component. Without this, a ref could write a downloaded file wherever it liked - refs
     * arrive from config files, server flags and scripts that interpolate variables, and "fetch a
     * URL to a path the input chooses" is not a property a downloader may have.
     */
    private static void component(String value, String what) {
        require(value != null && !value.isBlank(), "empty " + what);
        require(
                value.indexOf('/') < 0
                        && value.indexOf('\\') < 0
                        && value.indexOf(':') < 0
                        && value.indexOf('\0') < 0
                        && !value.equals(".")
                        && !value.equals(".."),
                what + " must be a single path component, not '" + value + "'");
    }

    /** The revision this ref means, explicit or the host's own. */
    String revisionOrDefault() {
        return revision == null ? host.defaultRevision : revision;
    }

    /** The quant this ref means. Null {@link #quant} also permits the single-file fallback. */
    String quantOrDefault() {
        return quant == null ? DEFAULT_QUANT : quant;
    }

    /** {@code owner/repo} - what an error should say when the file is not the point. */
    String repoId() {
        return owner + "/" + repo;
    }

    /**
     * The repository directory in the cache. The revision joins it only when the ref NAMED one, so
     * the common case stays byte-identical to the tree {@code scripts/download-models.sh} and the
     * test {@code ModelFixture} already populate.
     */
    String cacheRepo() {
        return revision == null ? repo : repo + "@" + revision;
    }

    /** The repository's web page, for a message that has to send someone to accept a licence. */
    String repoUrl() {
        return host.base() + host.prefix + "/" + repoId();
    }

    /** The download URL of {@code file}, repository-relative (a 302 to a signed CDN URL). */
    String fileUrl(String file) {
        return repoUrl() + "/resolve/" + revisionOrDefault() + "/" + file;
    }

    /** {@code location} joined onto {@code file}, the repository-relative path of a listed file. */
    String inLocation(String file) {
        return location.isEmpty() ? file : location + "/" + file;
    }

    @Override
    public String toString() {
        StringBuilder ref = new StringBuilder(host.name).append('/').append(repoId());
        if (revision != null) {
            ref.append('@').append(revision);
        }
        if (!location.isEmpty()) {
            ref.append('/').append(location);
        }
        if (quant != null) {
            ref.append(':').append(quant);
        }
        return ref.toString();
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
