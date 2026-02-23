package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.file.Paths;

/** Snippets for tensor access. */
@SuppressWarnings("unused")
public class TensorSnippets {

    // --8<-- [start:tensor-access]
    void listAllTensors() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        for (TensorEntry tensor : gguf.getTensors()) {
            System.out.println(tensor.name() + ": " + tensor.ggmlType());
        }
    }

    // --8<-- [end:tensor-access]

    // --8<-- [start:tensor-info]
    void getTensorInfo() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        TensorEntry tensor = gguf.getTensor("token_embd.weight");
        GGMLType type = tensor.ggmlType();
        long[] shape = tensor.shape();
        long size = tensor.byteSize();
        long offset = tensor.offset();
    }

    // --8<-- [end:tensor-info]

    // --8<-- [start:tensor-offset]
    void calculateOffset() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        TensorEntry tensor = gguf.getTensor("token_embd.weight");

        long absoluteOffset = tensor.absoluteOffset(gguf);
        // Or manually: gguf.getTensorDataOffset() + tensor.offset()
    }

    // --8<-- [end:tensor-offset]

    // --8<-- [start:tensor-contains]
    void checkTensorExists() throws IOException {
        GGUF gguf = GGUF.read(Paths.get("model.gguf"));
        if (gguf.containsTensor("output.weight")) {
            TensorEntry tensor = gguf.getTensor("output.weight");
        }
    }
    // --8<-- [end:tensor-contains]
}
