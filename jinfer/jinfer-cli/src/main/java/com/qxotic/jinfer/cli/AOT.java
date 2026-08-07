package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Timer;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.toknroll.Tokenizer;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32C;

/**
 * The native image's model preload ({@code -Djinfer.preload=<file>[<pathSeparator><file>...]} at
 * BUILD time): for each listed GGUF, everything derivable from its header - the parsed GGUF, and
 * the tokenizer where the header carries a vocabulary - is constructed at class-init and, since
 * {@code com.qxotic.*} initializes at image build, snapshots into the image heap. Both pieces are
 * pure heap; tensor data is still mmap'd at run time. Companions preload too: an mmproj or a
 * sidecar is just another header, without a vocabulary.
 *
 * <p>A preloaded header is used only when PROVEN to describe the candidate file's bytes, through
 * three layers of trust: the file SIZE gates for free - the only layer that saves work - then one
 * read of the header region ({@code [0, tensorDataOffset)} - exactly the bytes the parse consumed)
 * feeds CRC32C and SHA-256 in a single pass: two independent algorithms that must both agree, not a
 * cost ladder. Any mismatch warns and falls back to a fresh parse - the failure mode of a wrong
 * entry is a slower start, never a wrong header. Names carry no identity (every mmproj shares its
 * filename with every other one); they serve only the swapped-file warning.
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
            int headerCrc32c,
            String headerSha256,
            GGUF gguf,
            Tokenizer tokenizer) {}

    /** CRC32C and SHA-256 of one region, from the same single read pass. */
    record HeaderDigests(int crc32c, String sha256) {}

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
                GGUF gguf = ModelLoader.readGguf(fileChannel, path.toString());
                long headerLength = gguf.getTensorDataOffset();
                HeaderDigests digests = digestHeader(fileChannel, headerLength);
                Tokenizer tokenizer =
                        gguf.containsKey("tokenizer.ggml.tokens")
                                ? Tokenizers.fromGGUF(gguf)
                                : null; // an mmproj or a sidecar has no vocabulary
                baked.add(
                        new PreloadedFile(
                                path.getFileName().toString(),
                                fileChannel.size(),
                                headerLength,
                                digests.crc32c(),
                                digests.sha256(),
                                gguf,
                                tokenizer));
            } catch (IOException e) {
                throw new UncheckedIOException("cannot pre-load model: " + path, e);
            }
        }
        return List.copyOf(baked);
    }

    /** Package-visible for its test: both digests of {@code [0, length)}, one read pass. */
    static HeaderDigests digestHeader(FileChannel fileChannel, long length) throws IOException {
        CRC32C crc = new CRC32C();
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
            crc.update(buffer.duplicate());
            sha.update(buffer);
            position += read;
        }
        return new HeaderDigests((int) crc.getValue(), HexFormat.of().formatHex(sha.digest()));
    }

    /**
     * The preloaded file PROVEN to be {@code path}, or null. Size gates, then the header bytes must
     * digest identically - so a hit is a guarantee, and every kind of miss (unknown size, a size
     * tie, a content mismatch) warns where it helps and falls back to a fresh parse.
     * Package-visible, pure over {@code preloaded}, for its test.
     */
    static PreloadedFile match(List<PreloadedFile> preloaded, Path path) {
        long size;
        try {
            size = Files.size(path);
        } catch (IOException unreadable) {
            return null; // the real load will say what is wrong, with its own message
        }
        PreloadedFile found = null;
        for (PreloadedFile candidate : preloaded) {
            if (candidate.fileSize() == size) {
                if (found != null) {
                    LOG.log(
                            System.Logger.Level.WARNING,
                            "two preloaded files are both {0} bytes - cannot tell which one {1}"
                                    + " is, loading it fresh",
                            size,
                            path);
                    return null;
                }
                found = candidate;
            }
        }
        if (found == null) {
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
            HeaderDigests digests = digestHeader(fileChannel, found.headerLength());
            if (digests.crc32c() != found.headerCrc32c()
                    || !digests.sha256().equals(found.headerSha256())) {
                LOG.log(
                        System.Logger.Level.WARNING,
                        "{0} is the same size as the preloaded {1} but its header bytes differ -"
                                + " loading it fresh",
                        path,
                        found.fileName());
                return null;
            }
        } catch (IOException unreadable) {
            return null;
        }
        return found;
    }

    /**
     * The CLI's one load path. EVERY file consults its own preload independently - the main model
     * and each companion - so any subset of them being preloaded works, and a miss on one file
     * degrades only that file to a fresh parse: a preloaded mmproj keeps its baked header even
     * under a model the image has never seen. A runtime {@code -Djinfer.preTokenizer.*} override
     * makes the tokenizer rebuild so the override applies: the escape hatch outranks the preload,
     * and that precedence is this caller's policy, not the library's.
     */
    static LoadedModel<?> load(Path modelPath, Map<String, Path> companions) throws IOException {
        PreloadedFile main = match(PRELOADED, modelPath);
        Tokenizer tokenizer =
                main == null || Tokenizers.hasPropertyOverrides() ? null : main.tokenizer();
        Map<String, ModelProvider.Companion> attached = new LinkedHashMap<>();
        companions.forEach(
                (capability, path) -> {
                    PreloadedFile baked = match(PRELOADED, path);
                    attached.put(
                            capability,
                            new ModelProvider.Companion(path, baked == null ? null : baked.gguf()));
                });
        try (var timer = Timer.log("Load model");
                FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            GGUF gguf =
                    main != null
                            ? main.gguf()
                            : ModelLoader.readGguf(fileChannel, modelPath.toString());
            return Models.load(fileChannel, gguf, Arena.global(), attached, tokenizer);
        }
    }
}
