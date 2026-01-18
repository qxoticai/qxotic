package ai.qxotic.model.llm;

public abstract class AbstractHuggingFaceLoader<M extends Model<C, W, S>, C, W, S>
        implements ModelLoader<M, C, W, S> {

    protected final HuggingFaceModel huggingFace;

    protected AbstractHuggingFaceLoader(HuggingFaceModel huggingFace) {
        this.huggingFace = huggingFace;
    }
}
