package com.qxotic.jinfer.models.qwen3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * {@link ModelProvider} service for {@code general.architecture} "qwen3" - the RETRIEVAL family:
 * Qwen3-Embedding (pooled vectors) and Qwen3-Reranker (yes/no judge) over one backbone. The
 * generative Qwen 3.5 models are architecture "qwen35"/"qwen35moe", served by jinfer-qwen35.
 *
 * <p>Both GGUFs declare the same architecture, so the entry point the caller picks decides how the
 * weights are read: {@link com.qxotic.jinfer.chat.Models#loadEmbedder} pools, {@code loadReranker}
 * judges. Handing an embedding GGUF to the reranker (or the reverse) produces numbers, not an error
 * - it is the caller's file to choose. A caller that must choose for itself asks {@link
 * #retrieval}, which reads what the header does say: an embedding GGUF declares its pooling.
 */
public final class Qwen3Provider implements ModelProvider {

    @Override
    public Set<String> architectures() {
        return Set.of("qwen3");
    }

    /**
     * llama.cpp's converter writes {@code qwen3.pooling_type}: LAST (3) on Qwen3-Embedding, RANK
     * (4) on a reranker it converts. Older reranker conversions (mradermacher's) carry none, so an
     * absent key falls back to the name the converter stamped, and says nothing when that is silent
     * too - guessing would serve one face's numbers as the other's.
     */
    @Override
    public Optional<Retrieval> retrieval(GGUF gguf) {
        if (gguf.containsKey("qwen3.pooling_type"))
            return Optional.of(
                    gguf.getValue(int.class, "qwen3.pooling_type") == RANK_POOLING
                            ? Retrieval.RERANKING
                            : Retrieval.EMBEDDING);
        String name =
                (gguf.getStringOrDefault("general.name", "")
                                + " "
                                + gguf.getStringOrDefault("general.basename", ""))
                        .toLowerCase(Locale.ROOT);
        if (name.contains("rerank")) return Optional.of(Retrieval.RERANKING);
        if (name.contains("embed")) return Optional.of(Retrieval.EMBEDDING);
        return Optional.empty();
    }

    /** llama.cpp's {@code LLAMA_POOLING_TYPE_RANK}. */
    private static final int RANK_POOLING = 4;

    @Override
    public LoadedModel<?> loadLanguage(
            FileChannel fileChannel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer) {
        throw new IncompatibleModelException(
                "'qwen3' is the Qwen3 retrieval family (Qwen3-Embedding, Qwen3-Reranker), not a"
                        + " generative model; load it with Models.loadEmbedder or"
                        + " Models.loadReranker");
    }

    @Override
    public LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        Qwen3 model = Qwen3.loadModel(fileChannel, gguf, arena, tokenizer);
        // last-token pooling wants a trailing EOS on every sequence (the llama.cpp convention)
        int eos =
                SpecialTokens.find(model.tokenizer(), "<|endoftext|>")
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "qwen3 vocab has no <|endoftext|>"));
        return new LoadedEmbedder<>(
                model,
                model.tokenizer(),
                IntSequence.empty(),
                IntSequence.of(eos),
                model.configuration().embeddingLength(),
                32, // model card: Matryoshka output supports every width from 32 to native
                path.getFileName().toString(),
                // the card's instructed-query framing, default retrieval task, verbatim
                // (get_detailed_instruct: 'Instruct: {task}\nQuery:{query}' - no space after
                // Query:); documents are embedded bare per the same card
                "Instruct: Given a web search query, retrieve relevant passages that answer the"
                        + " query\nQuery:",
                "");
    }

    @Override
    public LoadedReranker<?> loadReranker(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        Qwen3 model = Qwen3.loadModel(fileChannel, gguf, arena, tokenizer);
        return new LoadedReranker<>(model, new Qwen3Reranker(model), path.getFileName().toString());
    }
}
