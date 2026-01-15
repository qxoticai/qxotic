package ai.qxotic.model.llm;

import ai.qxotic.model.llm.Model;
import ai.qxotic.format.safetensors.HuggingFace;

public abstract class AbstractHuggingFaceLoader<M extends Model<C, W, S>, C, W, S> implements ModelLoader<M, C, W, S> {

    protected final HuggingFace huggingFace;

    protected AbstractHuggingFaceLoader(HuggingFace huggingFace) {
        this.huggingFace = huggingFace;
    }
}
