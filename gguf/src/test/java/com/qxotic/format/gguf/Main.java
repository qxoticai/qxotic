package com.qxotic.format.gguf;

import java.io.IOException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.file.Path;

class Main {
    public static void main(String[] args) throws IOException {
        // https://github.com/ggml-org/llama.cpp/raw/refs/heads/master/models/ggml-vocab-phi-3.gguf");
        var url =
                new URL(
                        // "https://github.com/ggml-org/llama.cpp/blob/master/models/ggml-vocab-llama-bpe.gguf"
                        "https://hf.co/Qwen/Qwen3-235B-A22B-GGUF/resolve/main/Q8_0/Qwen3-235B-A22B-Q8_0-00001-of-00009.gguf"); // "https://huggingface.co/mukel/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true");

        var gguf = GGUF.read(Channels.newChannel(url.openStream()));
        Builder builder = Builder.newBuilder(gguf);
        for (TensorEntry tensorEntry : gguf.getTensors()) {
            String name = tensorEntry.name();
            builder.removeTensor(name);
        }
        for (String key : gguf.getMetadataKeys()) {
            if (!key.startsWith("tokenizer.")) {
                builder.removeKey(key);
            }
        }

        GGUF output = builder.build();
        GGUF.write(output, Path.of("tmp.gguf"));
        System.out.println(output.toString(true, true));
    }
}
