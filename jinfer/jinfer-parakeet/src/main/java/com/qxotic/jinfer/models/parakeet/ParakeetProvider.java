package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.chat.ModelProvider;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/** Architecture-dispatch entry for NVIDIA Parakeet: a transcription-only family. */
public final class ParakeetProvider implements ModelProvider {

    @Override
    public Set<String> architectures() {
        return Set.of(Parakeet.ARCHITECTURE);
    }

    @Override
    public TranscriptionModel<?, ?, ?> loadTranscription(
            FileChannel channel, GGUF gguf, Path path, Arena arena, Map<String, Path> companions)
            throws IOException {
        return Parakeet.load(channel, gguf, arena);
    }
}
