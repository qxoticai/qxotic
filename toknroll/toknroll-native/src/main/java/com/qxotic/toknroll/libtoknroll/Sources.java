package com.qxotic.toknroll.libtoknroll;

import com.qxotic.jinfer.hub.ModelRef;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
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
        Located at = locate(source, store);
        if (at.repository() != null)
            return at.ggufFile() != null
                    ? ggufFromHub(at.repository(), at.ggufFile())
                    : tiktokenFromHub(at.repository());
        if (!at.folder()) return read(at.path());
        if (!Files.isRegularFile(at.path().resolve("tokenizer.json")))
            throw Errors.notFound(at.path() + " holds no tokenizer.json"); // missing, as a file is
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
        if (!ModelStore.isRef(source)) { // a path is taken as given: a name may end in a space
            // a dotted first segment is a misspelled host, never an owner or a missing file
            if (!Files.exists(Path.of(source)) && source.matches("[^/\\\\:]+\\.[^/\\\\:]+/.+"))
                throw Errors.invalid(
                        "'"
                                + source
                                + "': unknown host; a hub is hf.co, huggingface.co or"
                                + " modelscope.cn");
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
            } catch (IllegalArgumentException missing) {
                // the store says what the folder holds: no folder at all (a cause), or a folder
                // without the file
                if (missing.getCause() != null && !ref.path().isEmpty())
                    return Located.file(store.resolve(source)); // a file, read whole
                if (ref.path().isEmpty() && missing.getMessage().contains("tiktoken.model"))
                    return Located.tiktoken(ref);
                throw Errors.notFound(missing.getMessage());
            }
        } catch (IllegalArgumentException e) {
            throw failure(e);
        } catch (UncheckedIOException e) {
            // offline, either way the store knows it: built without sources, or its own knob
            if (e.getMessage().contains("no source")
                    || e.getCause() instanceof FileNotFoundException)
                throw Errors.notFound(source + " is not cached, and offline mode never fetches");
            throw e;
        }
    }

    /**
     * The store reports a missing repository, folder, file or quant as an IllegalArgumentException
     * that says "no ..." (a 404 as its cause when a listing answered so); everything else it
     * refuses is the reference's spelling.
     */
    private static Errors.Failure failure(IllegalArgumentException e) {
        String message = e.getMessage() == null ? e.toString() : e.getMessage();
        return e.getCause() != null || message.startsWith("no ")
                ? Errors.notFound(message)
                : Errors.invalid(message);
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
            throw Errors.notFound(path + " does not exist");
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
        byte[] head;
        try (InputStream in = Files.newInputStream(file)) {
            head = in.readNBytes(64);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (hasGgufMagic(head)) return Kind.GGUF;
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
                                    + " folder holding one");
        };
    }

    /**
     * A hub GGUF, read for its header by toknroll's own bounded reader and cached under its root.
     */
    private static Tokenizer ggufFromHub(ModelRef ref, String file) {
        boolean offline = Stores.offline();
        return ref.host().equals("modelscope.cn")
                ? gguf().fromModelScope(
                                ref.owner(), ref.repo(), ref.revision(), file, offline, false)
                : gguf().fromHuggingFace(
                                ref.owner(), ref.repo(), ref.revision(), file, offline, false);
    }

    private static Tokenizer tiktokenFromHub(ModelRef ref) {
        boolean offline = Stores.offline();
        return ref.host().equals("modelscope.cn")
                ? HuggingFaceTokenizerLoader.fromModelScope(
                        ref.owner(), ref.repo(), ref.revision(), offline, false)
                : HuggingFaceTokenizerLoader.fromHuggingFace(
                        ref.owner(), ref.repo(), ref.revision(), offline, false);
    }

    /**
     * The GGUF loader, built at the first GGUF, the -Dtoknroll.gguf.* options read then: one they
     * break (an alias to no scheme) fails each GGUF load with its own message, and nothing else.
     */
    private static GGUFTokenizerLoader gguf() {
        GGUFTokenizerLoader loader = ggufLoader;
        if (loader != null) return loader;
        synchronized (Sources.class) {
            if (ggufLoader == null)
                ggufLoader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
            return ggufLoader;
        }
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
