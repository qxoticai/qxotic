package com.qxotic.model.llm.generic;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.AbstractGGUFLoader;
import com.qxotic.model.llm.ChatFormat;
import com.qxotic.model.llm.Loader;
import com.qxotic.model.llm.Model;
import com.qxotic.model.llm.ModelLoader;
import com.qxotic.model.llm.SpanLoader;
import com.qxotic.model.llm.apertus.ApertusChatFormat;
import com.qxotic.model.llm.gemma3.GemmaChatFormat;
import com.qxotic.model.llm.granite.GraniteChatFormat;
import com.qxotic.model.llm.llama.Llama3ChatFormat;
import com.qxotic.model.llm.mistral.MistralChatFormat;
import com.qxotic.model.llm.phi3.Phi3ChatFormat;
import com.qxotic.model.llm.qwen3.DeepSeekFormat;
import com.qxotic.toknroll.Tokenizer;
import java.util.stream.Stream;

public class GenericGGUFLoader
        extends AbstractGGUFLoader<Model<Object, Object, Object>, Object, Object, Object> {

    @Override
    public Model loadModel(Object configuration) {
        return modelLoader().loadModel(configuration);
    }

    private volatile ModelLoader modelLoader;

    ModelLoader modelLoader() {
        ModelLoader result = modelLoader;
        if (result == null) {
            synchronized (this) {
                result = modelLoader;
                if (result == null) {
                    modelLoader = result = new Loader().loadModelLoader(gguf);
                }
            }
        }
        return result;
    }

    GenericGGUFLoader(GGUF gguf) {
        super(gguf);
    }

    @Override
    public Object loadConfiguration(int maxTokens, SpanLoader spanLoader) {
        return modelLoader().loadConfiguration(maxTokens, spanLoader);
    }

    @Override
    public Object loadWeights(Object configuration, SpanLoader spanLoader) {
        return modelLoader().loadWeights(configuration, spanLoader);
    }

    private ChatFormat loadChatFormatHeuristic(Tokenizer tokenizer) {
        String chatTemplate = gguf.getValue(String.class, "tokenizer.chat_template");
        if (chatTemplate == null) {
            return new ChatMLFormat(tokenizer);
        }

        // Phi 3 +
        if (Stream.of("<|system|>", "<|assistant|>", "<|user|>", "<|end|>")
                .allMatch(chatTemplate::contains)) {
            return new Phi3ChatFormat(tokenizer);
        }

        // Mistral
        if (Stream.of("[INST]", "[/INST]").allMatch(chatTemplate::contains)) {
            return new MistralChatFormat(tokenizer);
        }

        // Llama 3 +
        if (Stream.of("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
                .allMatch(chatTemplate::contains)) {
            return new Llama3ChatFormat(tokenizer);
        }

        // Granite 3
        if (Stream.of("<|start_of_role|>", "<|end_of_role|>").allMatch(chatTemplate::contains)) {
            return new GraniteChatFormat(tokenizer);
        }

        // ChatML
        if (Stream.of("<|im_start|>", "<|im_end|>").allMatch(chatTemplate::contains)) {
            return new ChatMLFormat(tokenizer);
        }

        // DeepSeek distills (Qwen 3)
        if (Stream.of("<｜User｜>", "<｜Assistant｜>").allMatch(chatTemplate::contains)) {
            return new DeepSeekFormat(tokenizer);
        }

        // Gemma 3
        if (Stream.of("<start_of_turn>", "<end_of_turn>").allMatch(chatTemplate::contains)) {
            return new GemmaChatFormat(tokenizer);
        }

        if (Stream.of(
                        "<|system_start|>", "<|system_end|>",
                        "<|user_start|>", "<|user_end|>",
                        "<|assistant_start|>", "<|assistant_end|>")
                .allMatch(chatTemplate::contains)) {
            return new ApertusChatFormat(tokenizer);
        }

        return new ChatMLFormat(tokenizer);
    }

    @Override
    public ChatFormat createChatFormat(Tokenizer tokenizer) {
        return loadChatFormatHeuristic(tokenizer);
    }
}
