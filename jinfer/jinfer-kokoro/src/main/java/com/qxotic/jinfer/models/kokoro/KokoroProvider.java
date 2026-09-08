package com.qxotic.jinfer.models.kokoro;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/** Architecture-dispatch entry for Kokoro speech synthesis. */
public final class KokoroProvider implements ModelProvider {

    @Override
    public boolean supports(String architecture) {
        return Kokoro.ARCHITECTURE.equals(architecture);
    }

    @Override
    public Set<String> architectures() {
        return Set.of(Kokoro.ARCHITECTURE);
    }

    @Override
    public Map<String, String> companionFiles() {
        return Map.of("voice", "voice");
    }

    @Override
    public LoadedModel<?> load(
            FileChannel fileChannel,
            GGUF gguf,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer) {
        throw new UnsupportedOperationException(
                "'kokoro' is a speech-only family - load it with Models.loadSpeech");
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena) {
        throw new IllegalArgumentException("Kokoro requires the 'voice' companion GGUF");
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions)
            throws IOException {
        return loadSpeech(fileChannel, gguf, 0, arena, companions);
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel fileChannel,
            GGUF gguf,
            long baseOffset,
            Arena arena,
            Map<String, Path> companions)
            throws IOException {
        Path voice = companions.get("voice");
        if (voice == null)
            throw new IllegalArgumentException("Kokoro requires the 'voice' companion GGUF");
        return KokoroTTS.load(fileChannel, gguf, baseOffset, voice, arena);
    }
}
