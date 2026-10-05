package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Test-classpath only: real GGUF dispatch with weightless models, never a production architecture.
 */
public final class CliModelProvider implements ModelProvider {
    public static final Set<String> ARCHITECTURES =
            Set.of(
                    "cli_test_language",
                    "cli_test_speech",
                    "cli_test_transcription",
                    "cli_test_embedding",
                    "cli_test_reranker");
    static Arena weights;
    static Map<String, Path> attachments;
    static CliFixtures.Template template;
    static SpeakTest.Speech speech;
    static TranscribeTest.Transcriber transcription;
    static int languageLoads, speechLoads, transcriptionLoads, embedderLoads, rerankerLoads;

    static void reset() {
        weights = null;
        attachments = null;
        template = null;
        speech = null;
        transcription = null;
        languageLoads = speechLoads = transcriptionLoads = embedderLoads = rerankerLoads = 0;
    }

    public Set<String> architectures() {
        return ARCHITECTURES;
    }

    public Map<String, String> companionFiles() {
        return Map.of("media", "mmproj", "voice", "voice", "lexicon", "lexicon");
    }

    private static void record(GGUF gguf, Arena arena, Map<String, Path> companions)
            throws IOException {
        weights = arena;
        attachments = Map.copyOf(companions);
        if (gguf.getStringOrDefault("test.failure", "").equals("load"))
            throw new IOException("fixture load failed");
    }

    public LoadedModel<?> loadLanguage(
            FileChannel channel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        languageLoads++;
        if (!gguf.getString("general.architecture").equals("cli_test_language"))
            return ModelProvider.super.loadLanguage(
                    channel, gguf, path, arena, companions, tokenizer);
        record(gguf, arena, companions);
        template = new CliFixtures.Template();
        if (gguf.getStringOrDefault("test.failure", "").equals("bug")) {
            // a plain RuntimeException: the CLI reads the JDK's refusal types as the user's error
            template.failure =
                    new RuntimeException(
                            "fixture internal failure", new IOException("original cause"));
            template.failure.addSuppressed(new IOException("cleanup detail"));
        }
        return new LoadedModel<>(
                new TestLanguageModel(),
                tokenizer == null ? TestLanguageModel.TOKENIZER : tokenizer,
                "",
                Set.of(),
                new ContentKey("cli-workflow"),
                Optional.of(template),
                new LoadedModel.SamplingDefaults(0f, 1f, 0, 0f));
    }

    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        speechLoads++;
        if (!gguf.getString("general.architecture").equals("cli_test_speech"))
            return ModelProvider.super.loadSpeech(channel, gguf, path, arena, companions);
        record(gguf, arena, companions);
        speech = new SpeakTest.Speech();
        speech.fail = gguf.getStringOrDefault("test.failure", "").equals("generate");
        return speech;
    }

    /** The retrieval fixtures say their face from the header, as the real ports do. */
    public Optional<Retrieval> retrieval(GGUF gguf) {
        return switch (gguf.getString("general.architecture")) {
            case "cli_test_embedding" -> Optional.of(Retrieval.EMBEDDING);
            case "cli_test_reranker" -> Optional.of(Retrieval.RERANKING);
            default -> Optional.empty();
        };
    }

    /** The retrieval fixtures are claimed, never built: the server's own tests fake the models. */
    public LoadedEmbedder<?> loadEmbedder(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        embedderLoads++;
        if (!gguf.getString("general.architecture").equals("cli_test_embedding"))
            return ModelProvider.super.loadEmbedder(channel, gguf, path, arena, tokenizer);
        throw new IOException("the fixture embeds nothing");
    }

    public LoadedReranker<?> loadReranker(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        rerankerLoads++;
        if (!gguf.getString("general.architecture").equals("cli_test_reranker"))
            return ModelProvider.super.loadReranker(channel, gguf, path, arena, tokenizer);
        throw new IOException("the fixture ranks nothing");
    }

    public TranscriptionModel<?, ?, ?> loadTranscription(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        transcriptionLoads++;
        if (!gguf.getString("general.architecture").equals("cli_test_transcription"))
            return ModelProvider.super.loadTranscription(channel, gguf, path, arena, companions);
        record(gguf, arena, companions);
        transcription = new TranscribeTest.Transcriber();
        return transcription;
    }
}
