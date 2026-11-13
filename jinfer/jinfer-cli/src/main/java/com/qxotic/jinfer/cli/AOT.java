package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.BufferedInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * The native image's model preload ({@code -Djinfer.preload=<model>[<pathSeparator><model>...]} at
 * BUILD time): for each listed MODEL, everything derivable from its header - the parsed GGUF and
 * the tokenizer built from it - is constructed at class-init and, since {@code com.qxotic.*}
 * initializes at image build, snapshots into the image heap. Both pieces are pure heap; tensor data
 * is still mmap'd at run time. MODELS only: a companion (mmproj, sidecar, lexicon) is parsed by its
 * port at load in ~10 ms - nothing worth a bake, so listing one is refused with the reason rather
 * than baked into dead image bytes.
 *
 * <p>TO ITS CALLERS THIS CLASS IS A TRANSPARENT CACHE IN FRONT OF {@link Models}: each method
 * mirrors a path-based Models entry, forwards to it, and adds no behaviour - same validation, same
 * errors, same results, observably different only in speed. The CLI's rule is that path-based
 * Models entries the preload can serve are called through here; nothing else in the CLI knows the
 * preload exists.
 *
 * <p>A preloaded header is used only when PROVEN to describe the candidate file's bytes: the file
 * SIZE gates for free, then the SHA-256 of the header region ({@code [0, tensorDataOffset)} -
 * exactly the bytes the parse consumed) must match. Any mismatch warns and falls back to a fresh
 * parse - the failure mode of a wrong entry is a slower start, never a wrong header. Names carry no
 * identity (every mmproj shares its filename with every other one); they serve only the
 * swapped-file warning.
 *
 * <p>The model and encoder OBJECTS are not preloaded, BY DESIGN, not as a gap: headers and
 * tokenizers are pure heap and derive from bytes alone, while the objects wire into mmap'd segments
 * - baking them would put segment handles in the image heap and demand a per-port build/attach
 * split, to save ~150 ms of honest construction. Mmap plus the tokenizer is the endpoint.
 */
final class AOT {

    private static final System.Logger LOG = System.getLogger("jinfer.load");

    record PreloadedFile(
            String fileName,
            long fileSize,
            long headerLength,
            String headerSha256,
            GGUF gguf,
            Tokenizer tokenizer) {}

    private static final List<PreloadedFile> PRELOADED =
            preload(System.getProperty("jinfer.preload"));

    static List<PreloadedFile> preload(String paths) {
        if (paths == null || paths.isEmpty()) {
            return List.of();
        }
        List<PreloadedFile> baked = new ArrayList<>();
        for (String entry : paths.split(File.pathSeparator)) {
            Path path = Path.of(entry.strip());
            if (!Files.isRegularFile(path)) {
                throw new IllegalArgumentException("cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = readGguf(fileChannel);
                if (!gguf.containsKey("tokenizer.ggml.tokens")) {
                    throw new IllegalArgumentException(
                            path
                                    + " has no vocabulary, so there is nothing worth baking: the"
                                    + " preload serves text models, whose cost is the vocabulary"
                                    + " build. Companions (mmproj, sidecars) and vocabulary-less"
                                    + " models are parsed at load, cheaply.");
                }
                long headerLength = gguf.getTensorDataOffset();
                baked.add(
                        new PreloadedFile(
                                path.getFileName().toString(),
                                fileChannel.size(),
                                headerLength,
                                headerSha256(fileChannel, headerLength),
                                gguf,
                                tokenizerFrom(gguf)));
            } catch (IOException e) {
                throw new UncheckedIOException("cannot pre-load model: " + path, e);
            }
        }
        return List.copyOf(baked);
    }

    /** The header parse, exactly as {@link Models} does it on its own loads. */
    private static GGUF readGguf(FileChannel fileChannel) throws IOException {
        fileChannel.position(0L);
        return GGUF.read(
                Channels.newChannel(
                        new BufferedInputStream(Channels.newInputStream(fileChannel), 1 << 20)));
    }

    /** The tokenizer described by a model header. */
    private static Tokenizer tokenizerFrom(GGUF gguf) {
        return GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
    }

    /** Package-visible for its test: the SHA-256 of {@code [0, length)}. */
    static String headerSha256(FileChannel fileChannel, long length) throws IOException {
        MessageDigest sha;
        try {
            sha = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e); // SHA-256 is mandatory in every JRE
        }
        ByteBuffer buffer = ByteBuffer.allocate(1 << 20);
        long position = 0;
        while (position < length) {
            buffer.clear().limit((int) Math.min(buffer.capacity(), length - position));
            int read = fileChannel.read(buffer, position);
            if (read <= 0) {
                throw new EOFException("header ends at " + position + " of " + length);
            }
            buffer.flip();
            sha.update(buffer);
            position += read;
        }
        return HexFormat.of().formatHex(sha.digest());
    }

    /**
     * The preloaded file PROVEN to be {@code path}, or null. Size gates, then the header bytes must
     * digest identically - so a hit is a guarantee, and every kind of miss (unknown size, a content
     * mismatch) warns where it helps and falls back to a fresh parse. Package-visible, pure over
     * {@code preloaded}, for its test.
     */
    static PreloadedFile match(List<PreloadedFile> preloaded, Path path) {
        long size;
        try {
            size = Files.size(path);
        } catch (IOException unreadable) {
            return null; // the real load will say what is wrong, with its own message
        }
        List<PreloadedFile> sameSize =
                preloaded.stream().filter(candidate -> candidate.fileSize() == size).toList();
        if (sameSize.isEmpty()) {
            for (PreloadedFile candidate : preloaded) {
                // a NAME match with a size mismatch is a swapped or re-downloaded file: the
                // preload silently missing here would just look like a slow start
                if (candidate.fileName().equals(path.getFileName().toString())) {
                    LOG.log(
                            System.Logger.Level.WARNING,
                            "{0} is not the preloaded {1}: expected {2} bytes but the file is {3}"
                                    + " - loading it fresh",
                            path,
                            candidate.fileName(),
                            candidate.fileSize(),
                            size);
                    break;
                }
            }
            return null;
        }
        try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
            for (PreloadedFile candidate : sameSize) {
                if (headerSha256(fileChannel, candidate.headerLength())
                        .equals(candidate.headerSha256())) {
                    return candidate;
                }
            }
        } catch (IOException unreadable) {
            return null;
        }
        LOG.log(
                System.Logger.Level.WARNING,
                "{0} is the same size as the preloaded {1} but its header bytes differ - loading"
                        + " it fresh",
                path,
                sameSize.getFirst().fileName());
        return null;
    }

    /**
     * The capabilities {@code modelPath}'s architecture offers - from the preloaded header when the
     * match proves one, else one fresh header read. The CLI's {@code --with} validation asks here
     * so a preloaded model is not re-parsed just to check a flag.
     */
    static Map<String, String> companionFiles(Path modelPath) throws IOException {
        PreloadedFile main = match(PRELOADED, modelPath);
        return main != null ? Models.companionFiles(main.gguf()) : Models.companionFiles(modelPath);
    }

    /**
     * The tokenizer another GGUF describes ({@code --with tokenizer=...}): served from the bake
     * when the file is a proven preload, built fresh otherwise.
     */
    static Tokenizer tokenizerFrom(Path path) throws IOException {
        PreloadedFile baked = match(PRELOADED, path);
        if (baked != null) {
            return baked.tokenizer();
        }
        try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
            return tokenizerFrom(readGguf(fileChannel));
        } catch (IllegalArgumentException | EOFException e) {
            // the reader names the fault, not the file; a truncated file is not a GGUF either
            throw new IllegalArgumentException(
                    "cannot load tokenizer from '" + path + "': " + e.getMessage(), e);
        }
    }

    /**
     * The CLI's one load path: the model consults the preload, companions pass through untouched
     * (their ports parse them), and {@code tokenizerOverride} - the file named by {@code --with
     * tokenizer=} - outranks everything when present. Otherwise the baked tokenizer serves. That
     * precedence - explicit file, then bake, then the GGUF's own - is this caller's policy, not the
     * library's. The caller owns the cross-thread weights arena and closes it after the engine.
     */
    static LoadedModel<?> load(
            Path modelPath, Map<String, Path> companions, Path tokenizerOverride, Arena arena)
            throws IOException {
        PreloadedFile main = match(PRELOADED, modelPath);
        Tokenizer tokenizer =
                tokenizerOverride != null
                        ? tokenizerFrom(tokenizerOverride)
                        : main == null ? null : main.tokenizer();
        return Models.load(modelPath, arena, companions, tokenizer);
    }
}
