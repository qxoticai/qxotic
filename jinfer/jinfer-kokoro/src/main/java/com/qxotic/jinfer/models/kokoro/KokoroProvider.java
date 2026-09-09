package com.qxotic.jinfer.models.kokoro;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/** Architecture-dispatch entry for Kokoro speech synthesis: a speech-only family. */
public final class KokoroProvider implements ModelProvider {

    @Override
    public Set<String> architectures() {
        return Set.of(Kokoro.ARCHITECTURE);
    }

    @Override
    public Map<String, String> companionFiles() {
        return Map.of("voice", "voice");
    }

    @Override
    public SpeechSynthesisModel<?, ?, ?> loadSpeech(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        Path voice = companions.get("voice");
        if (voice == null)
            throw new IllegalArgumentException("Kokoro requires the 'voice' companion GGUF");
        return KokoroTTS.load(channel, gguf, voice, arena);
    }
}
