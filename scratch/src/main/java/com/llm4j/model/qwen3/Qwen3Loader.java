package com.llm4j.model.qwen3;

import com.llm4j.gguf.GGUF;
import com.llm4j.model.SpanLoader;
import com.llm4j.model.llama.BaseLlamaLoader;
import com.llm4j.model.llama.DefaultKernelOps;
import com.llm4j.model.llama.Llama;
import com.llm4j.model.qwen2.Qwen2TextSplitterFactory;
import com.llm4j.tokenizers.Tokenizer;
import com.llm4j.tokenizers.impl.Tiktoken;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.regex.Pattern;


public class Qwen3Loader extends BaseLlamaLoader {

    static final String QWEN3_ARCH = "qwen3";

    public Qwen3Loader(GGUF gguf) {
        super(gguf);
    }

    @Override
    public Qwen3 loadModel(Llama.Configuration configuration) {
        return new Qwen3(configuration, DefaultKernelOps.getKernelOps(), DefaultKernelOps.getSpanFactory());
    }

    @Override
    public Qwen3.Configuration loadConfiguration(int maxTokens, SpanLoader spanLoader) {
        String arch = gguf.getValue(String.class, "general.architecture");
        if (!QWEN3_ARCH.equals(arch)) {
            throw new IllegalArgumentException("general.architecture expected " + QWEN3_ARCH + " but found " + arch);
        }
        return super.loadConfiguration(maxTokens, spanLoader).with(b -> b.ropeIsNeoxStyle(true));
    }

    public static Tokenizer loadTokenizerFromTiktoken(Path tiktokenModel) throws IOException {
        try (var reader = Files.newBufferedReader(tiktokenModel)) {
            Map<String, Integer> mergeableRanks = Tiktoken.loadMergeableRanks(reader);
            return loadTokenizerFromTiktoken(mergeableRanks);
        }
    }

    static Map<String, Integer> qwen2SpecialTokens() {
        return Map.ofEntries(
                Map.entry("<|endoftext|>", 151643),
                Map.entry("<|im_start|>", 151644),
                Map.entry("<|im_end|>", 151645),
                Map.entry("<|object_ref_start|>", 151646),
                Map.entry("<|object_ref_end|>", 151647),
                Map.entry("<|box_start|>", 151648),
                Map.entry("<|box_end|>", 151649),
                Map.entry("<|quad_start|>", 151650),
                Map.entry("<|quad_end|>", 151651),
                Map.entry("<|vision_start|>", 151652),
                Map.entry("<|vision_end|>", 151653),
                Map.entry("<|vision_pad|>", 151654),
                Map.entry("<|image_pad|>", 151655),
                Map.entry("<|video_pad|>", 151656),
                // These are not marked as special in HF.
                Map.entry("<tool_call>", 151657),
                Map.entry("</tool_call>", 151658),
                Map.entry("<|fim_prefix|>", 151659),
                Map.entry("<|fim_middle|>", 151660),
                Map.entry("<|fim_suffix|>", 151661),
                Map.entry("<|fim_pad|>", 151662),
                Map.entry("<|repo_name|>", 151663),
                Map.entry("<|file_sep|>", 151664)
        );
    }

    public static Tokenizer loadTokenizerFromTiktoken(Map<String, Integer> mergeableRanks) {
        return Tiktoken.createFromTiktoken("qwen2", mergeableRanks, Pattern.compile(Qwen2TextSplitterFactory.QWEN2_PATTERN), qwen2SpecialTokens());
    }
}
