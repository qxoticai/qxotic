package com.qxotic.jinfer.x.models.qwen3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.chat.LoadedEmbedder;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.chat.LoadedReranker;
import com.qxotic.jinfer.x.chat.ModelProvider;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * {@link ModelProvider} service for {@code general.architecture} "qwen3" - the RETRIEVAL family:
 * Qwen3-Embedding (pooled vectors) and Qwen3-Reranker (yes/no judge) over one backbone. The
 * generative Qwen 3.5 models are architecture "qwen35"/"qwen35moe", served by jinfer-xqwen35.
 *
 * <p>Both GGUFs declare the same architecture and cannot be told apart from metadata, so the entry
 * point the caller picks decides how the weights are read: {@link
 * com.qxotic.jinfer.x.chat.Models#loadEmbedder} pools, {@code loadReranker} judges. Handing an
 * embedding GGUF to the reranker (or the reverse) produces numbers, not an error - it is the
 * caller's file to choose.
 */
public final class Qwen3Provider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return "qwen3".equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of("qwen3");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer) {
        throw new UnsupportedOperationException(
                "'qwen3' is the Qwen3 RETRIEVAL family (Qwen3-Embedding, Qwen3-Reranker), not a"
                        + " generative model - load it with Models.loadEmbedder or"
                        + " Models.loadReranker");
    }

    @Override
    public LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) throws IOException {
        Qwen3 model = Qwen3.loadModel(fileChannel, gguf, arena);
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
                new int[0],
                new int[] {eos},
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
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) throws IOException {
        Qwen3 model = Qwen3.loadModel(fileChannel, gguf, arena);
        return new LoadedReranker<>(model, new Qwen3Reranker(model), path.getFileName().toString());
    }
}
