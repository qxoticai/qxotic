package com.llm4j.model;

import com.llm4j.api.model.Model;
import com.llm4j.gguf.GGUF;
import com.llm4j.gguf.TensorInfo;
import com.llm4j.huggingface.HFTensorEntry;
import com.llm4j.huggingface.HuggingFace;

public abstract class AbstractGGUFLoader<M extends Model<C, W, S>, C, W, S> implements ModelLoader<M, C, W, S> {

    protected final GGUF gguf;

    protected AbstractGGUFLoader(GGUF gguf) {
        this.gguf = gguf;
    }
}

