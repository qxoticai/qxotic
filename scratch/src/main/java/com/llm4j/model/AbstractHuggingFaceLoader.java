package com.llm4j.model;

import com.llm4j.api.model.Model;
import com.llm4j.huggingface.HFTensorEntry;
import com.llm4j.huggingface.HuggingFace;

public abstract class AbstractHuggingFaceLoader<M extends Model<C, W, S>, C, W, S> implements ModelLoader<M, C, W, S> {

    protected final HuggingFace huggingFace;

    protected AbstractHuggingFaceLoader(HuggingFace huggingFace) {
        this.huggingFace = huggingFace;
    }
}
