package ai.qxotic.format.safetensors;

import ai.qxotic.format.safetensors.impl.JSON;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

public class HuggingFace {

    protected final Path rootPath;

    static final String MODEL_CONFIG = "config.json";
    static final String MODEL_SAFETENSORS = "model.safetensors";
    static final String SAFETENSORS_INDEX = "model.safetensors.index.json";

    private final Map<String, HFTensorEntry> tensors;

    public HuggingFace(Path rootPath) {
        this.rootPath = rootPath;
        this.tensors = loadTensorEntries();
    }

    public Path getRootPath() {
        return rootPath;
    }

    public Map<String, Object> loadModelConfig() {
        try {
            return (Map<String, Object>) JSON.parse(Files.readString(rootPath.resolve(MODEL_CONFIG)));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public Map<String, HFTensorEntry> loadTensorEntries() {
        try {
            Path indexPath = rootPath.resolve(SAFETENSORS_INDEX);
            if (Files.exists(indexPath)) {
                return Safetensors.loadFromModelRoot(rootPath);
            } else {
                return Safetensors.loadTensorEntries(rootPath.resolve(MODEL_SAFETENSORS));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public Map<String, HFTensorEntry> getTensors() {
        return Collections.unmodifiableMap(tensors);
    }

    public HFTensorEntry getTensor(String tensorName) {
        return tensors.get(tensorName);
    }

    public boolean containsTensor(String tensorName) {
        return tensors.containsKey(tensorName);
    }
}
