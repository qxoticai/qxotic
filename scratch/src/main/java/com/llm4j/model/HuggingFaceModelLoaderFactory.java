package com.llm4j.model;

import com.llm4j.huggingface.HuggingFace;

public interface HuggingFaceModelLoaderFactory extends ModelLoaderFactory<HuggingFace> {
    @Override
    default String getFormatName() {
        return "huggingface";
    }
}
