package com.qxotic.model.llm;

import com.qxotic.format.gguf.GGUF;

public abstract class AbstractGGUFLoader<M extends Model<C, W, S>, C, W, S>
        implements ModelLoader<M, C, W, S> {

    protected final GGUF gguf;

    protected AbstractGGUFLoader(GGUF gguf) {
        this.gguf = gguf;
    }
}
