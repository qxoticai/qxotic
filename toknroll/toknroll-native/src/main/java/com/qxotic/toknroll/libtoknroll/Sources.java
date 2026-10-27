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
 * toknroll_load's sources, as toknroll.h specifies them, in three steps: resolve the source to a
 * file or folder (jinfer-hub's grammar and store), sniff a file's content for its format, load with
 * that format's loader. A hub GGUF is read for its header only, by toknroll-gguf's own bounded
 * reader; a repository shipping tiktoken.model goes through toknroll-hf's own hub path, which also
 * fetches the pattern module such repositories carry.
 */
final class Sources {
    private static final GGUFTokenizerLoader GGUF_LOADER =
            GGUFTokenizerLoader.createBuilderWithBuiltins().build();
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
        return at.folder() ? HuggingFaceTokenizerLoader.fromLocal(at.path()) : read(at.path());
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
        source = source.strip();
        if (source.matches("(?i)(hf|ms|huggingface|modelscope):.*")) // the prefixes of 0.3.0
        throw Errors.invalid(
                    "'"
                            + source
                            + "': a hub is named by its host, hf.co/owner/repo or"
                            + " modelscope.cn/owner/repo");
        if (!ModelStore.isRef(source)) {
            // a dotted first segment is a misspelled host, never an owner or a missing file
            if (!Files.exists(Path.of(source)) && source.matches("[^/\\\\:]+\\.[^/\\\\:]+/.+"))
                throw Errors.invalid(
                        "'"
                                + source
                                + "': unknown host; a hub is hf.co, huggingface.co or"
                                + " modelscope.cn");
            return local(source);
        }
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
            case GGUF -> GGUF_LOADER.fromLocal(file);
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
                ? GGUF_LOADER.fromModelScope(
                        ref.owner(), ref.repo(), ref.revision(), file, offline, false)
                : GGUF_LOADER.fromHuggingFace(
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

    private static boolean hasGgufMagic(byte[] bytes) {
        return bytes.length >= GGUF_MAGIC.length
                && Arrays.equals(Arrays.copyOf(bytes, GGUF_MAGIC.length), GGUF_MAGIC);
    }

    private static boolean isGguf(String path) {
        return path.toLowerCase(Locale.ROOT).endsWith(".gguf");
    }

    private Sources() {}
}
