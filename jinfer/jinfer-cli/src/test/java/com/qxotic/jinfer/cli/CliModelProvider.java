package com.qxotic.jinfer.cli;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.chat.LoadedModel;
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
            Set.of("cli_test_language", "cli_test_speech", "cli_test_transcription");
    static Arena weights;
    static Map<String, Path> attachments;
    static CliFixtures.Template template;
    static SpeakTest.Speech speech;
    static TranscribeTest.Transcriber transcription;
    static int languageLoads, speechLoads, transcriptionLoads;

    static void reset() {
        weights = null;
        attachments = null;
        template = null;
        speech = null;
        transcription = null;
        languageLoads = speechLoads = transcriptionLoads = 0;
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
            template.failure =
                    new IllegalStateException(
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
