package com.qxotic.jinfer.x.models.lfm2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.x.chat.LoadedEmbedder;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.chat.LoadedReranker;
import com.qxotic.jinfer.x.chat.ModelProvider;
import com.qxotic.jinfer.x.chat.Models;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** {@link ModelProvider} service: the Lfm2 port's arch-dispatch entry. */
public final class Lfm2Provider implements ModelProvider {

  @Override
  public boolean supports(String architecture) {
    return architecture.startsWith("lfm");
  }

  @Override
  public Set<String> architectures() {
    return Set.of("lfm2", "lfm2moe"); // representative: supports() matches lfm*
  }

  @Override
  public Map<String, String> companionFiles() {
    return Map.of("media", "mmproj");
  }

  @Override
  public LoadedModel<?> load(
      FileChannel fileChannel,
      GGUF gguf,
      Arena arena,
      Map<String, Path> companions,
      Tokenizer tokenizer)
      throws IOException {
    Lfm2 model = Lfm2.loadModel(fileChannel, gguf, arena, tokenizer);
    Path media = companions.get("media");
    if (media != null) model = model.withMedia(media, arena);
    // a retrieval checkpoint "loads" as a chat model and then generates noise - refuse by name
    if (!model.config().causalAttention()) {
      throw new IllegalArgumentException(
          "this LFM2 checkpoint is a RETRIEVAL model (non-causal attention), not a"
              + " generative one - it belongs to an embedder/reranker load path,"
              + " not Models.load");
    }
    Tokenizer tok = model.tokenizer();
    String source = gguf.getStringOrDefault("tokenizer.chat_template", "");
    // the 2.6B-era template's generation prompt is "<|im_start|>assistant\n<think>" - detect
    // the pre-opened think span from the checkpoint's own template source
    boolean opensThink = source.contains("assistant\n<think>");
    return new LoadedModel<>(
        model,
        tok,
        source,
        SpecialTokens.stops(tok, -1, "<|im_end|>", "<eos>", "<|endoftext|>", "<end_of_turn>"),
        Models.modelSeed(fileChannel),
        Optional.of(new Lfm2ChatTemplate(model, opensThink)),
        LoadedModel.SamplingDefaults.NONE);
  }

  /**
   * The LFM2.5-Embedding checkpoints: same architecture string as the generative models, told apart
   * by their OWN metadata (non-causal attention + CLS pooling) - so a generative GGUF handed here
   * refuses loudly instead of producing numbers.
   */
  @Override
  public LoadedEmbedder<?> loadEmbedder(FileChannel fileChannel, GGUF gguf, Path path, Arena arena)
      throws IOException {
    Lfm2 model = Lfm2.loadModel(fileChannel, gguf, arena, null);
    if (!model.config().isEmbedder())
      throw new IllegalArgumentException(
          path.getFileName()
              + " is a generative LFM2 checkpoint, not an embedder (embedders"
              + " declare pooling_type and non-causal attention) - chat with it via"
              + " Models.load, or embed with LFM2.5-Embedding-350M-GGUF");
    // CLS pooling reads the BOS row, so every sequence leads with it (add_bos in the GGUF)
    int bos = gguf.getValue(int.class, "tokenizer.ggml.bos_token_id");
    return new LoadedEmbedder<Lfm2.State>(
        model,
        model.tokenizer(),
        new int[] {bos},
        new int[0],
        model.config().embeddingLength(),
        model.config().embeddingLength(), // fixed 1024-dim vector; no MRL claim
        path.getFileName().toString(),
        // the card's retrieval framing: LFM2.5-Embedding is trained on this exact pair
        "query: ",
        "document: ");
  }

  /**
   * The family's reranker is LFM2.5-ColBERT: late interaction (MaxSim over per-token {@code
   * dense_2} embeddings), not a cross-encoder judge. Detected by its own metadata, so a wrong file
   * refuses by name instead of producing numbers.
   */
  @Override
  public LoadedReranker<?> loadReranker(FileChannel fileChannel, GGUF gguf, Path path, Arena arena)
      throws IOException {
    Lfm2 model = Lfm2.loadModel(fileChannel, gguf, arena, null);
    if (!model.config().isColbert() || model.weights().dense2() == null)
      throw new IllegalArgumentException(
          path.getFileName()
              + " is not the family's reranker - LFM2 reranking is"
              + " LFM2.5-ColBERT-350M-GGUF (per-token dense_2 embeddings, MaxSim);"
              + " a generative checkpoint chats via Models.load, the embedding"
              + " checkpoint embeds via Models.loadEmbedder");
    int bos = gguf.getValue(int.class, "tokenizer.ggml.bos_token_id");
    int pad = gguf.getValue(int.class, "tokenizer.ggml.padding_token_id");
    return new LoadedReranker<>(
        model, new Lfm2Colbert(model, bos, pad), path.getFileName().toString());
  }
}
