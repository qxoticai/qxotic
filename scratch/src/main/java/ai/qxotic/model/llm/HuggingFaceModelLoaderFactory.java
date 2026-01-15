package ai.qxotic.model.llm;

import ai.qxotic.format.safetensors.HuggingFace;

public interface HuggingFaceModelLoaderFactory extends ModelLoaderFactory<HuggingFace> {
    @Override
    default String getFormatName() {
        return "huggingface";
    }
}
