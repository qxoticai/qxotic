package com.qxotic.toknroll.libtoknroll;

import com.qxotic.jinfer.hub.ModelNotFoundException;
import com.qxotic.jinfer.hub.ModelRef;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.TokenizerLoadException;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.Optional;

/**
 * toknroll_tokenizer_load's sources, as toknroll.h specifies them, in three steps: resolve the
 * source to a file or folder (jinfer-hub's grammar and store), sniff a file's content for its
 * format, load with that format's loader. A hub GGUF is read for its header only, by
 * toknroll-gguf's own bounded reader; a repository shipping tiktoken.model goes through
 * toknroll-hf's own hub path, which also fetches the pattern module such repositories carry.
 */
final class Sources {
    private static volatile GGUFTokenizerLoader ggufLoader;
    private static final byte[] GGUF_MAGIC = {'G', 'G', 'U', 'F'};

    static Tokenizer load(String source) {
        return load(source, Stores.store());
    }

    static Tokenizer load(String source, ModelStore store) {
        try {
            return load(source, locate(source, store));
        } catch (Errors.Failure e) {
            // a bare reference with a path spells a relative path too, and the file would have
            // won had it existed: the hub's "not found" gets the hint the unknown-host error gives
            if (e.code == Errors.ENOTFOUND
                    && ModelStore.isRef(source)
                    && source.strip().matches("[^/\\\\.]+/[^/\\\\]+/.+"))
                throw Errors.notFound(
                        e.getMessage() + "; a local path is written './" + source.strip() + "'");
            throw e;
        }
    }

    /** The source as written names the fetch's failures, as the store's own do. */
    private static Tokenizer load(String source, Located at) {
        if (at.repository() != null) return fromHub(source, at);
        if (!at.folder()) {
            try {
                return read(at.path());
            } catch (RuntimeException e) {
                // a hub file the cache holds is never refetched over a bad copy: the reader's error
                // says which file, and this says what to do; a valid file toknroll does not
                // implement
                // is no bad copy, and stays Unsupported, the caller's cue to fall back
                if (!ModelStore.isRef(source)
                        || (e instanceof Errors.Failure f && f.code != Errors.EINVAL)
                        || Errors.code(e) == Errors.EUNSUPPORTED) throw e;
                throw Errors.invalid(
                        e.getMessage() + "; the cache's copy: delete it to fetch it again");
            }
        }
        // a folder holds a tokenizer.json, or a tiktoken.model read as a hub checkout is; neither
        // is missing, as a file is; one that cannot be listed is not missing either
        if (!Files.isReadable(at.path()))
            throw new Errors.Failure(Errors.ERROR, at.path() + ": permission denied");
        if (!Files.isRegularFile(at.path().resolve("tokenizer.json"))
                && !Files.isRegularFile(at.path().resolve("tiktoken.model")))
            throw Errors.notFound(
                    at.path() + " is a folder that holds no tokenizer.json nor tiktoken.model");
        return HuggingFaceTokenizerLoader.fromLocal(at.path());
    }

    /**
     * What a source resolves to: a file or folder on disk, or something the tokenizer loaders fetch
     * themselves from a repository: a GGUF read for its header only, or a tiktoken repository.
     */
    record Located(Path path, boolean folder, ModelRef repository, String ggufFile) {
        static Located file(Path path) {
            return new Located(path, false, null, null);
        }

        static Located folder(Path path) {
            return new Located(path, true, null, null);
        }

        static Located gguf(ModelRef repository, String file) {
            return new Located(null, false, repository, file);
        }

        static Located tiktoken(ModelRef repository) {
            return new Located(null, false, repository, null);
        }
    }

    static Located locate(String source, ModelStore store) {
        if (source == null || source.isBlank()) throw Errors.invalid("source is empty");
        if (source.contains("://")) // a browser's URL, pasted: the grammar has no scheme
        throw Errors.invalid("'" + source + "' is a URL, not a source: write [host/]owner/repo");
        if (!ModelStore.isRef(source)) { // a path is taken as given: a name may end in a space
            // a dotted first segment is a misspelled host, never an owner or a missing file
            if (!Files.exists(Path.of(source)) && source.matches("[^/\\\\:]+\\.[^/\\\\:]+/.+"))
                throw Errors.invalid(
                        "'"
                                + source
                                + "': unknown host; a hub is hf.co, huggingface.co or"
                                + " modelscope.cn, and a local path is written './"
                                + source
                                + "'");
            return local(source);
        }
        source = source.strip(); // a ref is lenient, as ModelRef's own parse is
        ModelRef ref;
        try {
            ref = ModelRef.parse(source);
        } catch (IllegalArgumentException e) {
            throw Errors.invalid(e.getMessage());
        }
        try {
            if (ref.quant() != null || isGguf(ref.path())) {
                Optional<Path> whole = store.find(source); // a model jinfer has is read in place
                if (whole.isPresent()) return Located.file(whole.get());
                return Located.gguf(ref, ref.quant() != null ? store.select(source) : ref.path());
            }
            if (!ref.path().isEmpty()) { // a file the cache holds is read in place, as a GGUF is
                Optional<Path> held = store.find(source);
                if (held.isPresent() && Files.isRegularFile(held.get()))
                    return Located.file(held.get());
            }
            try {
                return Located.file(store.resolve(source + "/tokenizer.json"));
            } catch (ModelNotFoundException missing) {
                // offline, a bare repository may be a tiktoken one, whose files live in the
                // loader's own cache: its cache-only path reads them, or says not cached
                if (Stores.offline()) {
                    if (ref.path().isEmpty()) return Located.tiktoken(ref);
                    throw missing;
                }
                // online the store says what the folder holds: no folder at all (a 404, the
                // cause), or a folder without the file
                if (missing.getCause() != null && !ref.path().isEmpty())
                    return Located.file(store.resolve(source)); // a file, read whole
                if (ref.path().isEmpty() && missing.getMessage().contains("tiktoken.model"))
                    return Located.tiktoken(ref);
                throw missing;
            }
        } catch (ModelNotFoundException e) {
            throw Errors.notFound(e.getMessage());
        } catch (IllegalArgumentException e) {
            throw Errors.invalid(e.getMessage());
        } catch (UncheckedIOException e) {
            // a store built without sources: offline by construction, so a bare repository may
            // still be a tiktoken one the loader has cached
            if (e.getMessage().contains("no source")) {
                if (ref.quant() == null && ref.path().isEmpty()) return Located.tiktoken(ref);
                throw notCached(source);
            }
            throw e;
        }
    }

    private static Errors.Failure notCached(String source) {
        return Errors.notFound(source + " is not cached, and offline mode never fetches");
    }

    private static Located local(String source) {
        Path path;
        try {
            path = Path.of(source);
        } catch (InvalidPathException e) {
            throw Errors.invalid(e.getMessage());
        }
        if (!Files.exists(path)) {
            if (source.indexOf('/') < 0 && source.indexOf('\\') < 0) // nor a reference
            throw Errors.invalid(
                        "'"
                                + source
                                + "' is neither a file here nor a reference:"
                                + " [host/]owner/repo[@revision][/path][:quant]");
            // a C string reaches no shell: ~ is a folder's name, as fopen reads it
            String home = source.startsWith("~") ? " (~ is not expanded: write the full path)" : "";
            throw Errors.notFound(
                    source + " does not exist" + home); // as written: no \ for a / on Windows
        }
        return Files.isDirectory(path) ? Located.folder(path) : Located.file(path);
    }

    enum Kind {
        GGUF,
        JSON,
        UNKNOWN
    }

    /** The file's content decides: a GGUF by its magic, a tokenizer.json by being a JSON object. */
    static Kind sniff(Path file) {
        // a FIFO, a socket or a device would block or stream: only a regular file is opened
        if (!Files.isRegularFile(file)) throw Errors.invalid(file + " is not a regular file");
        byte[] head;
        try (InputStream in = Files.newInputStream(file)) {
            head = in.readNBytes(64);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (hasGgufMagic(head)) return Kind.GGUF;
        // JSON may not start with one (RFC 8259); Hugging Face and Python's json refuse it too
        if (head.length >= 3
                && (head[0] & 0xFF) == 0xEF
                && (head[1] & 0xFF) == 0xBB
                && (head[2] & 0xFF) == 0xBF)
            throw Errors.invalid(
                    file
                            + " starts with a UTF-8 byte order mark, which JSON does not allow:"
                            + " save it without one");
        for (byte b : head) {
            if (b == '{') return Kind.JSON;
            if (!Character.isWhitespace(b)) break;
        }
        return Kind.UNKNOWN;
    }

    private static Tokenizer read(Path file) {
        return switch (sniff(file)) {
            case GGUF -> gguf().fromLocal(file);
            case JSON -> HuggingFaceTokenizerLoader.fromLocal(file);
            case UNKNOWN ->
                    throw Errors.invalid(
                            file
                                    + " is not a tokenizer: expected a GGUF, a tokenizer.json, or a"
                                    + " folder holding one or a tiktoken.model");
        };
    }

    /**
     * What the loaders fetch themselves: a hub GGUF, read for its header by toknroll's own bounded
     * reader, or a tiktoken repository, each cached under the loader's root.
     */
    private static Tokenizer fromHub(String source, Located at) {
        ModelRef ref = at.repository();
        boolean offline = Stores.offline();
        boolean modelscope = ref.host().equals("modelscope.cn");
        try {
            if (at.ggufFile() != null)
                return modelscope
                        ? gguf().fromModelScope(
                                        ref.owner(),
                                        ref.repo(),
                                        ref.revision(),
                                        at.ggufFile(),
                                        offline,
                                        false)
                        : gguf().fromHuggingFace(
                                        ref.owner(),
                                        ref.repo(),
                                        ref.revision(),
                                        at.ggufFile(),
                                        offline,
                                        false);
            return modelscope
                    ? HuggingFaceTokenizerLoader.fromModelScope(
                            ref.owner(), ref.repo(), ref.revision(), offline, false)
                    : HuggingFaceTokenizerLoader.fromHuggingFace(
                            ref.owner(), ref.repo(), ref.revision(), offline, false);
        } catch (TokenizerLoadException e) {
            throw failure(source, ref, offline, e);
        }
    }

    /**
     * A loader's own fetch failure in the header's codes. Offline, the transport refuses every
     * request as a NoSuchFileException, and a cache folder the loader cannot make holds nothing
     * either: not cached. Online, a 401, 403 or 404 is a repository or file the hub does not serve,
     * said as the store says it for a listing, and a cache folder the loader cannot make is said in
     * words. Anything else is the loader's own failure, as it says it.
     */
    static RuntimeException failure(
            String source, ModelRef ref, boolean offline, TokenizerLoadException e) {
        Throwable cause = e.getCause();
        if (offline
                && (cause instanceof NoSuchFileException || cause instanceof AccessDeniedException))
            return notCached(source);
        if (cause instanceof AccessDeniedException denied)
            return new Errors.Failure(
                    Errors.ERROR, "cache root is not writable: " + denied.getFile());
        // the loaders' caches say "(HTTP <status>)", a 404 as a FileNotFoundException
        if (cause instanceof FileNotFoundException
                || (cause instanceof IOException
                        && String.valueOf(cause.getMessage()).matches(".*\\(HTTP 40[134]\\)")))
            return Errors.notFound(denied(ref));
        return e;
    }

    /**
     * What a 401, 403 or 404 means to a user, as jinfer-hub says it for a listing: Hugging Face
     * answers 401 for a repository that does not exist too. The loaders read a token from HF_TOKEN
     * or MODELSCOPE_TOKEN, and nowhere else.
     */
    private static String denied(ModelRef ref) {
        boolean modelscope = ref.host().equals("modelscope.cn");
        return ref.repoId()
                + " does not exist, or is gated or private: check the name, or accept its licence"
                + " at "
                + (modelscope ? "https://modelscope.cn/models/" : "https://huggingface.co/")
                + ref.repoId()
                + " and set "
                + (modelscope ? "MODELSCOPE_TOKEN" : "HF_TOKEN");
    }

    /**
     * The GGUF loader, built at the first GGUF, the -Dtoknroll.gguf.* options read then: one they
     * break (an alias to no scheme) fails each GGUF load with its own message, and nothing else. No
     * lock, as {@link Stores#store}: threads that race build equal loaders and one is kept, and a
     * build cut short leaves nothing held for the next.
     */
    private static GGUFTokenizerLoader gguf() {
        GGUFTokenizerLoader loader = ggufLoader;
        if (loader == null)
            ggufLoader = loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        return loader;
    }

    private static boolean hasGgufMagic(byte[] bytes) {
        return bytes.length >= GGUF_MAGIC.length
                && Arrays.equals(Arrays.copyOf(bytes, GGUF_MAGIC.length), GGUF_MAGIC);
    }

    private static boolean isGguf(String path) {
        return path.toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private Sources() {}
}
