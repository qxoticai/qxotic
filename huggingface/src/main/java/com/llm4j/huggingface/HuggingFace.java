package com.llm4j.huggingface;

import com.llm4j.huggingface.impl.JSON;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public class HuggingFace {

    public Path getRootPath() {
        return rootPath;
    }

    protected final Path rootPath;

    static final String MODEL_CONFIG = "config.json";
    static final String MODEL_SAFETENSORS = "model.safetensors";
    static final String SAFETENSROS_INDEX = "model.safetensors.index.json";

    private final Map<String, HFTensorEntry> tensors;

    public HuggingFace(Path rootPath) {
        this.rootPath = rootPath;
        this.tensors = loadTensorEntries();
    }

    public Map<String, Object> loadModelConfig() {
        try {
            return (Map<String, Object>) JSON.parse(Files.readString(rootPath.resolve(MODEL_CONFIG)));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public Map<String, HFTensorEntry> loadTensorEntries() {
        // TODO(peterssen): Support sharding.
        Map<String, HFTensorEntry> tensorEntries = null;
        try {
            tensorEntries = SafeTensors.loadTensorEntries(rootPath.resolve(MODEL_SAFETENSORS));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return tensorEntries;
    }

    public HFTensorEntry getTensor(String tensorName) {
        return tensors.get(tensorName);
    }

    public boolean containsTensor(String tensorName) {
        return tensors.containsKey(tensorName);
    }
}
