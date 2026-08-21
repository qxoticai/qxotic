package com.qxotic.jinfer.hub;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * A model reference. One exact grammar, no scheme, no query, no aliases, no browser views:
 *
 * <pre>
 *   host / owner/repo [@revision] [/path] [:quant]
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
 * <p>NOTHING here knows about file formats. Whether a path names a file or a folder is answered by
 * the repository listing, not by an extension. What jinfer is willing to LOAD is a separate
 * question, answered once in {@link ModelStore}.
 */
public record ModelRef(
        String host, String owner, String repo, String revision, String path, String quant) {

    /**
     * A host jinfer can talk to as a repository. The row is the one place a host is declared: its
     * canonical ref name plus the transport details {@link RepositorySource} uses (base URL,
     * prefix, token and mirror variables, default branch). Adding a host is one enum constant.
     */
    enum Host {
        HF("hf.co", "main", "HF_TOKEN", "HF_ENDPOINT", "https://huggingface.co", ""),
        MODELSCOPE(
                "modelscope.cn",
                "master",
                "MODELSCOPE_API_TOKEN",
                "MODELSCOPE_ENDPOINT",
                "https://modelscope.cn",
                "/models");

        final String name;
        final String defaultRevision;
        final String tokenEnv;
        final String endpointEnv;
        final String defaultBase;
        final String prefix;

        Host(
                String name,
                String defaultRevision,
                String tokenEnv,
                String endpointEnv,
                String defaultBase,
                String prefix) {
            this.name = name;
            this.defaultRevision = defaultRevision;
            this.tokenEnv = tokenEnv;
            this.endpointEnv = endpointEnv;
            this.defaultBase = defaultBase;
            this.prefix = prefix;
        }

        /** The base URL, honoring this host's mirror variable ({@code HF_ENDPOINT} etc.). */
        String base() {
            String endpoint = System.getenv(endpointEnv);
            if (endpoint == null || endpoint.isBlank()) {
                return defaultBase;
            }
            String trimmed = endpoint.strip();
            return trimmed.endsWith("/") ? trimmed.substring(0, trimmed.length() - 1) : trimmed;
        }
    }

    /** The host this exact name spells, or null. Only the canonical names are known. */
    private static Host lookup(String name) {
        String canonical = name.toLowerCase(Locale.ROOT);
        for (Host host : Host.values()) {
            if (host.name.equals(canonical)) {
                return host;
            }
        }
        return null;
    }

    /** Whether this is a host a ref may name - the first segment of every repository ref. */
    static boolean isKnownHost(String name) {
        return lookup(name) != null;
    }

    public ModelRef {
        component(host, "host");
        component(owner, "owner");
        component(repo, "repository");
        if (revision != null) {
            component(revision, "revision");
        }
        if (quant != null) {
            component(quant, "quant");
        }
        for (String segment : segments(path)) {
            component(segment, "path segment");
        }
    }

    /**
     * Whether {@code candidate} names a model ref: its first segment is a known host. The one rule
     * separating remote from local, consulted against a table rather than the filesystem.
     */
    static boolean isRef(String candidate) {
        if (candidate == null || candidate.isBlank()) {
            return false;
        }
        String rest = candidate.strip();
        int slash = rest.indexOf('/');
        return slash > 0 && lookup(rest.substring(0, slash)) != null;
    }

    /** Parses a ref, per the grammar in the class note. */
    public static ModelRef parse(String ref) {
        require(ref != null && !ref.isBlank(), "empty model ref");
        String rest = ref.strip();
        int slash = rest.indexOf('/');
        Host host = slash > 0 ? lookup(rest.substring(0, slash)) : null;
        require(host != null, shape(ref));
        rest = rest.substring(slash + 1);
        while (rest.endsWith("/")) {
            rest = rest.substring(0, rest.length() - 1);
        }
        List<String> parts = segments(rest);
        require(parts.size() >= 2, shape(ref));

        // the quant is a colon in the FINAL segment
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
        int at = repo.indexOf('@');
        if (at >= 0) {
            revision = repo.substring(at + 1);
            repo = repo.substring(0, at);
        }
        List<String> path = parts.subList(2, parts.size());
        return new ModelRef(host.name, owner, repo, revision, String.join("/", path), quant);
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
                + "'. A model ref names its host: hf.co/owner/repo[:quant] or"
                + " modelscope.cn/owner/repo[:quant], for example"
                + " hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M.";
    }

    /**
     * Every field becomes a directory or file name under the cache root, so every field must be ONE
     * path component. Without this, a ref could write a downloaded file wherever it liked.
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

    /** This ref's host as a protocol row, or null when the host names nothing jinfer ships. */
    Host hostKind() {
        return lookup(host);
    }

    /** The revision this ref means, explicit or the host's own. */
    String revisionOrDefault() {
        Host kind = hostKind();
        require(kind != null, "unknown host: " + host);
        return revision == null ? kind.defaultRevision : revision;
    }

    /** {@code owner/repo} - what an error should say when the file is not the point. */
    public String repoId() {
        return owner + "/" + repo;
    }

    /** The repository directory in the cache; a named revision joins the repo directory. */
    String cacheRepo() {
        return revision == null ? repo : repo + "@" + revision;
    }

    /** The repository's web page, for a message that has to send someone to accept a licence. */
    String repoUrl() {
        Host kind = hostKind();
        require(kind != null, "unknown host: " + host);
        return kind.base() + kind.prefix + "/" + repoId();
    }

    @Override
    public String toString() {
        StringBuilder ref = new StringBuilder(host).append('/').append(repoId());
        if (revision != null) {
            ref.append('@').append(revision);
        }
        if (!path.isEmpty()) {
            ref.append('/').append(path);
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
