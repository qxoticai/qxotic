package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Timer;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The native image's model preload ({@code -Djinfer.PreloadGGUF=<file>[,<file>...]} at BUILD time):
 * for each listed GGUF, everything derivable from its header - the parsed GGUF, and the tokenizer
 * where the header carries a vocabulary - is constructed at class-init and, since {@code
 * com.qxotic.*} initializes at image build, snapshots into the image heap. Both pieces are pure
 * heap; tensor data is still mmap'd at run time. Companions preload too: an mmproj or a sidecar is
 * just another header, without a vocabulary.
 *
 * <p>Matching is by FILE SIZE, never by name - companion files (mmproj, sidecars) all carry the
 * same few names, so a name is no identity at all. A miss or an ambiguity falls back to a fresh
 * parse, so a wrong preload entry can never be used through a MISS; what a size check cannot
 * exclude is a same-size different-bytes file.
 *
 * <p>The model and encoder OBJECTS are not preloaded, BY DESIGN, not as a gap: headers and
 * tokenizers are pure heap and derive from bytes alone, while the objects wire into mmap'd segments
 * - baking them would put segment handles in the image heap and demand a per-port build/attach
 * split, to save ~150 ms of honest construction. Mmap plus the tokenizer is the endpoint.
 */
final class AOT {

    private static final System.Logger LOG = System.getLogger("jinfer.load");

    // ponytail: file size is the whole fingerprint - cheap and name-independent, but a same-size
    // different-bytes file would match wrongly. The upgrade path is a header hash (bake the
    // tensor-data offset and a digest of the header bytes; verify before use). The NAME is kept
    // for diagnostics only: a name match with a size mismatch is a swapped file, worth a warning.
    record PreloadedFile(String fileName, long fileSize, GGUF gguf, Tokenizer tokenizer) {}

    private static final List<PreloadedFile> PRELOADED =
            preload(System.getProperty("jinfer.PreloadGGUF"));

    private static List<PreloadedFile> preload(String paths) {
        if (paths == null || paths.isEmpty()) {
            return List.of();
        }
        List<PreloadedFile> baked = new ArrayList<>();
        for (String entry : paths.split(",")) {
            Path path = Path.of(entry.strip());
            if (!Files.isRegularFile(path)) {
                throw new IllegalArgumentException("cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, path.toString());
                Tokenizer tokenizer =
                        gguf.containsKey("tokenizer.ggml.tokens")
                                ? Tokenizers.fromGGUF(gguf)
                                : null; // an mmproj or a sidecar has no vocabulary
                baked.add(
                        new PreloadedFile(
                                path.getFileName().toString(),
                                fileChannel.size(),
                                gguf,
                                tokenizer));
            } catch (IOException e) {
                throw new UncheckedIOException("cannot pre-load model: " + path, e);
            }
        }
        return List.copyOf(baked);
    }

    /**
     * The one preloaded file of exactly this size, or null - on a tie, nobody wins (fresh parse).
     */
    private static PreloadedFile match(Path path) {
        long size;
        try {
            size = Files.size(path);
        } catch (IOException unreadable) {
            return null; // the real load will say what is wrong, with its own message
        }
        PreloadedFile found = null;
        for (PreloadedFile candidate : PRELOADED) {
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
            for (PreloadedFile candidate : PRELOADED) {
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
        PreloadedFile main = match(modelPath);
        Tokenizer tokenizer =
                main == null || Tokenizers.hasPropertyOverrides() ? null : main.tokenizer();
        Map<String, ModelProvider.Companion> attached = new LinkedHashMap<>();
        companions.forEach(
                (capability, path) -> {
                    PreloadedFile baked = match(path);
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
