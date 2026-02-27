package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.file.Paths;

/** Snippets for building and modifying GGUF files. */
@SuppressWarnings("unused")
public class BuildingSnippets {

    // --8<-- [start:builder-create]
    void createNewGGUF() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "model")
                        .putInteger("llama.context_length", 4096)
                        .putTensor(
                                TensorEntry.create(
                                        "weights", new long[] {1024, 1024}, GGMLType.F32, 0L))
                        .build();
    }

    // --8<-- [end:builder-create]

    // --8<-- [start:builder-modify]
    void modifyExisting() throws IOException {
        GGUF original = GGUF.read(Paths.get("model.gguf"));
        GGUF modified =
                Builder.newBuilder(original)
                        .putString("general.version", "2.0")
                        .removeKey("deprecated.key")
                        .build(false); // Preserve tensor offsets
    }

    // --8<-- [end:builder-modify]

    // --8<-- [start:builder-alignment]
    void setAlignment() {
        Builder builder = Builder.newBuilder().setAlignment(64);
        int alignment = builder.getAlignment();
    }

    // --8<-- [end:builder-alignment]

    // --8<-- [start:modify-metadata]
    void updateMetadata() throws IOException {
        GGUF original = GGUF.read(Paths.get("model.gguf"));
        GGUF modified =
                Builder.newBuilder(original)
                        .putString("general.version", "2.0")
                        .putString("general.description", "Fine-tuned")
                        .removeKey("deprecated.key")
                        .build(false);
    }

    // --8<-- [end:modify-metadata]

    // --8<-- [start:check-alignment]
    void checkAlignment() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        TensorEntry tensor = gguf.getTensor("weights");
        long offset = gguf.absoluteOffset(tensor);
        int alignment = gguf.getAlignment();
        boolean isAligned = (offset % alignment) == 0;
    }
    // --8<-- [end:check-alignment]
}
