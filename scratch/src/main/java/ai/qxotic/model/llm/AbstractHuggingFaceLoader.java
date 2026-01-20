package ai.qxotic.model.llm;

import ai.qxotic.format.safetensors.SafetensorsIndex;

public abstract class AbstractHuggingFaceLoader<M extends Model<C, W, S>, C, W, S>
        implements ModelLoader<M, C, W, S> {

    protected final SafetensorsIndex safetensorsIndex;

    protected AbstractHuggingFaceLoader(SafetensorsIndex safetensorsIndex) {
        this.safetensorsIndex = safetensorsIndex;
    }
}
