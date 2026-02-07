package ai.qxotic.model.llm;

public interface HuggingFaceModelLoaderFactory extends ModelLoaderFactory<Object> {
    @Override
    default String getFormatName() {
        return "huggingface";
    }
}
