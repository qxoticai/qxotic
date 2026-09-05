package com.qxotic.jinfer.hub;

import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * A model reference: one string that names a model in a remote repository.
 *
 * <pre>
 *   [host/]owner/repo[@revision][/path][:quant]
 *
 *   unsloth/Qwen3.5-4B-GGUF                                              the default quant at the root
 *   unsloth/Qwen3.5-4B-GGUF:Q8_0                                         pick a quant
 *   LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf             pick a file: the vision projector
 *   unsloth/gemma-4-E2B-it-GGUF/mtp-gemma-4-E2B-it.gguf                  or the MTP draft head
 *   unsloth/gemma-4-E2B-it-GGUF/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf         a file in a subfolder
 *   unsloth/Qwen3.5-4B-GGUF@a1b2c3d:Q8_0                                 pin a revision (branch, tag, commit)
 *   hf.co/unsloth/Qwen3.5-4B-GGUF:Q8_0                                   the default host, written out
 *   modelscope.cn/unsloth/Qwen3.5-4B-GGUF:Q8_0                           ModelScope, the other host
 * </pre>
 *
 * <p>Position is the whole grammar: {@code /path} says WHERE, {@code :quant} says WHICH, {@code
 * @revision} says WHEN. A ref is never a URL - no scheme, no query, no {@code /blob/} browser
 * view. The host table ({@link Host}) is closed and the default is hardcoded, so one string still
 * means one thing on every machine, in a script and in a config file alike.
 *
 * <p>The host is optional and defaults to {@link Host#HF}, matching what llama.cpp's {@code -hf}
 * and {@code from_pretrained} already accept. Naming a host is how a ref reaches any OTHER source:
 * {@code modelscope.cn/owner/repo}. An explicit host always wins over the default.
 *
 * <p>Three rules keep a host-less ref from swallowing a local path, so no working path changes
 * meaning by gaining a default host: a file that EXISTS under that name wins, anything spelling a
 * path ({@code ./x}, {@code /x}, {@code ~/x}, a backslash, a drive letter) is a path, and {@code
 * owner/model.gguf} is a path too, because no repository is named after a model file. A dot in the
 * first segment reserves it for the host table, so a misspelled host is reported, never read as an
 * owner.
 *
 * <p>Whether a path names a file or a folder is answered by the repository listing, never by an
 * extension. What jinfer will LOAD is {@link ModelStore}'s call, not this record's.
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

        /** The host a ref names ({@link ModelRef#host()}), or null for a host this enum lacks. */
        static Host byName(String name) {
            for (Host host : values()) if (host.name.equals(name)) return host;
            return null;
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
     * Whether {@code candidate} names a model ref. Naming a known host makes one outright; a
     * host-less {@code owner/repo} makes one too, unless a file of that name already exists, in
     * which case the file wins - no working local path changes meaning by gaining a default host.
     */
    static boolean isRef(String candidate) {
        return isHostQualified(candidate)
                || (isBareRefShape(candidate) && !existsAsFile(candidate));
    }

    /** Whether the first segment spells a host the closed table knows. */
    static boolean isHostQualified(String candidate) {
        String first = firstSegment(candidate);
        return first != null && lookup(first) != null;
    }

    /**
     * Whether the first segment is SPELLED like a host, whether or not the table knows it. A dot
     * reserves the segment: {@code example.org/owner/repo} is a misspelled host, never an owner
     * called {@code example.org}. Without the rule a typo in a host silently becomes a repository
     * nobody owns, and its 404 is the first hint of it.
     */
    static boolean namesAHost(String candidate) {
        String first = firstSegment(candidate);
        return first != null && first.indexOf('.') >= 0;
    }

    /**
     * Whether {@code candidate} is shaped like a host-less {@code owner/repo[...]}: two segments or
     * more, spelling neither a URL, nor a filesystem path, nor a host. What each SEGMENT may
     * contain is the record's business, checked once in its constructor.
     */
    static boolean isBareRefShape(String candidate) {
        if (candidate == null || candidate.isBlank()) {
            return false;
        }
        String rest = candidate.strip();
        if (rest.contains("://") || isPathShape(rest) || namesAHost(rest)) {
            return false;
        }
        List<String> parts = segments(rest);
        // owner/repo names a REPOSITORY, and no repository is called model.gguf: a two-segment
        // string ending in a model extension is a relative path meant for modelPath(...). Three
        // segments or more is the file-in-repository form (owner/repo/mmproj-F32.gguf), untouched.
        return parts.size() >= 2 && !(parts.size() == 2 && namesAModelFile(parts.get(1)));
    }

    /** A file someone means to load, never the name of a repository. */
    private static boolean namesAModelFile(String segment) {
        String lower = segment.toLowerCase(Locale.ROOT);
        return lower.endsWith(".gguf") || lower.endsWith(".safetensors");
    }

    /**
     * Whether the string spells a path on this machine rather than a repository: a leading {@code
     * /}, {@code .} or {@code ~}, any backslash, or a drive letter. The one place that rule lives.
     */
    static boolean isPathShape(String value) {
        return value.startsWith("/")
                || value.startsWith(".")
                || value.startsWith("~")
                || value.indexOf('\\') >= 0
                || (value.length() >= 2
                        && Character.isLetter(value.charAt(0))
                        && value.charAt(1) == ':');
    }

    /** The text before the first {@code /}, or null when the string has no segments to split. */
    private static String firstSegment(String candidate) {
        if (candidate == null || candidate.isBlank()) {
            return null;
        }
        String rest = candidate.strip();
        int slash = rest.indexOf('/');
        return slash > 0 ? rest.substring(0, slash) : null;
    }

    /** Whether this string already names a file here. An unparseable path is simply not one. */
    private static boolean existsAsFile(String candidate) {
        try {
            return Files.exists(Path.of(candidate.strip()));
        } catch (InvalidPathException notAPath) {
            return false;
        }
    }

    /** Parses a ref, per the grammar in the class note. */
    public static ModelRef parse(String ref) {
        require(ref != null && !ref.isBlank(), "empty model ref");
        String rest = ref.strip();
        String first = firstSegment(rest);
        Host host = first == null ? null : lookup(first);
        if (host != null) {
            rest = rest.substring(first.length() + 1);
        } else {
            require(isBareRefShape(rest), shape(ref)); // host-less: hf.co carries it
            host = Host.HF;
        }
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
                + "'. A model ref is owner/repo[:quant], for example"
                + " unsloth/gemma-4-E2B-it-GGUF:Q4_K_M, which hf.co carries. Name a host to reach"
                + " another source: hf.co/owner/repo[:quant] or"
                + " modelscope.cn/owner/repo[:quant].";
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
