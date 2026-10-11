package com.qxotic.jinfer.models.gemmaembedding2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** {@link ModelProvider} service for {@code general.architecture} "gemma-embedding2". */
public final class GemmaEmbedding2Provider implements ModelProvider {

    @Override
    public Set<String> architectures() {
        return Set.of(GemmaEmbedding2.ARCHITECTURE);
    }

    @Override
    public Optional<Retrieval> retrieval(GGUF gguf) {
        return Optional.of(Retrieval.EMBEDDING);
    }

    /** The mmproj's image, audio and video towers. */
    @Override
    public Map<String, String> companionFiles() {
        return Map.of("media", "mmproj");
    }

    @Override
    public LoadedEmbedder<?> loadEmbedder(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        return loadEmbedder(channel, gguf, path, arena, Map.of(), tokenizer);
    }

    /**
     * Every sequence is framed {@code <bos> ... <eos>} (the GGUF's add_bos/add_eos); the card's
     * retrieval prefixes and Matryoshka widths down to 128; a {@code "media"} companion attaches
     * the mmproj's towers.
     */
    @Override
    public LoadedEmbedder<?> loadEmbedder(
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        GemmaEmbedding2 model = GemmaEmbedding2.loadModel(channel, gguf, arena, tokenizer);
        Path mmproj = companions.get("media");
        if (mmproj != null) model = model.withMedia(mmproj, arena);
        return new LoadedEmbedder<>(
                model,
                model.tokenizer(),
                IntSequence.of(model.bos),
                IntSequence.of(model.eos),
                model.configuration().embeddingLengthOut(),
                128,
                path.getFileName().toString(),
                "task: search result | query: ",
                "title: none | text: ");
    }
}
