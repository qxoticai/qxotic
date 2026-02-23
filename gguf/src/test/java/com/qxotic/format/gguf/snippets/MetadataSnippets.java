package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.GGUF;
import java.io.IOException;
import java.nio.file.Paths;

/** Snippets for metadata access. */
@SuppressWarnings("unused")
public class MetadataSnippets {

    // --8<-- [start:basic-info]
    void basicInfo() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        long tensorDataOffset = gguf.getTensorDataOffset();
        int alignment = gguf.getAlignment();
        boolean hasTensor = gguf.containsTensor("token_embd.weight");
    }

    // --8<-- [end:basic-info]

    // --8<-- [start:metadata-access]
    void accessMetadata() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        String name = gguf.getValue(String.class, "general.name");
        int contextLength = gguf.getValue(int.class, "llama.context_length");
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
    }

    // --8<-- [end:metadata-access]

    // --8<-- [start:metadata-or-default]
    void accessWithDefault() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        int contextLength = gguf.getValueOrDefault(int.class, "llama.context_length", 4096);
    }

    // --8<-- [end:metadata-or-default]

    // --8<-- [start:metadata-check]
    void checkKeyExists() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        if (gguf.containsKey("general.author")) {
            String author = gguf.getValue(String.class, "general.author");
        }
    }

    // --8<-- [end:metadata-check]

    // --8<-- [start:metadata-keys]
    void listKeys() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        for (String key : gguf.getMetadataKeys()) {
            System.out.println(key);
        }
    }

    // --8<-- [end:metadata-keys]

    // --8<-- [start:metadata-arrays]
    void accessArrays() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        String[] vocab = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        float[] scores = gguf.getValue(float[].class, "tokenizer.ggml.scores");
    }
    // --8<-- [end:metadata-arrays]
}
