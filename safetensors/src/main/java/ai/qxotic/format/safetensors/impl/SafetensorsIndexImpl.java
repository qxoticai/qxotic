package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.json.JSON;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.SafetensorsFormatException;
import ai.qxotic.format.safetensors.SafetensorsIndex;
import ai.qxotic.format.safetensors.TensorEntry;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class SafetensorsIndexImpl implements SafetensorsIndex {
    private static final String MODEL_SAFETENSORS = "model.safetensors";
    private static final String SAFETENSORS_INDEX = "model.safetensors.index.json";

    private final Path rootPath;
    private final Map<String, Path> tensorIndex;

    private SafetensorsIndexImpl(Path rootPath, Map<String, Path> tensorIndex) {
        this.rootPath = rootPath;
        this.tensorIndex = Collections.unmodifiableMap(tensorIndex);
    }

    @Override
    public Path getRootPath() {
        return rootPath;
    }

    @Override
    public Path getSafetensorsPath(String tensorName) {
        return tensorIndex.get(tensorName);
    }

    @Override
    public Collection<String> getTensorNames() {
        return tensorIndex.keySet();
    }

    public static SafetensorsIndex load(Path path) throws IOException {
        Map<String, Path> tensorIndex = new LinkedHashMap<>();
        Path rootPath;

        if (Files.isDirectory(path)) {
            rootPath = path;
            Path indexPath = path.resolve(SAFETENSORS_INDEX);

            if (Files.exists(indexPath)) {
                loadSharded(path, indexPath, tensorIndex);
            } else {
                Path singleFile = path.resolve(MODEL_SAFETENSORS);
                if (!Files.exists(singleFile)) {
                    throw new IOException(
                            "No safetensors files found (expected "
                                    + MODEL_SAFETENSORS
                                    + " or "
                                    + SAFETENSORS_INDEX
                                    + ")");
                }
                loadSingleFile(singleFile, tensorIndex);
            }
        } else if (Files.isRegularFile(path) && path.toString().endsWith(".safetensors")) {
            rootPath =
                    path.getParent() != null ? path.getParent() : path.toAbsolutePath().getParent();
            loadSingleFile(path, tensorIndex);
        } else {
            throw new IOException("Path must be a directory or .safetensors file: " + path);
        }

        return new SafetensorsIndexImpl(rootPath, tensorIndex);
    }

    private static void loadSharded(Path rootPath, Path indexPath, Map<String, Path> tensorIndex)
            throws IOException {
        Map<String, Object> index;
        try {
            index = (Map<String, Object>) JSON.parse(Files.readString(indexPath));
        } catch (JSON.ParseException e) {
            throw new SafetensorsFormatException(
                    "Invalid JSON in " + SAFETENSORS_INDEX + ": " + e.getMessage(), e);
        }

        // Validate and extract weight_map
        Object weightMapObj = index.get("weight_map");
        if (weightMapObj == null) {
            throw new SafetensorsFormatException("Missing 'weight_map' in " + SAFETENSORS_INDEX);
        }
        if (!(weightMapObj instanceof Map)) {
            throw new SafetensorsFormatException(
                    "'weight_map' must be an object in "
                            + SAFETENSORS_INDEX
                            + ", got "
                            + weightMapObj.getClass().getSimpleName());
        }

        Map<?, ?> weightMapRaw = (Map<?, ?>) weightMapObj;
        for (Map.Entry<?, ?> entry : weightMapRaw.entrySet()) {
            if (!(entry.getKey() instanceof String)) {
                throw new SafetensorsFormatException(
                        "'weight_map' keys must be strings in " + SAFETENSORS_INDEX);
            }
            if (!(entry.getValue() instanceof String)) {
                throw new SafetensorsFormatException(
                        "'weight_map' values must be strings in " + SAFETENSORS_INDEX);
            }

            String tensorName = (String) entry.getKey();
            String fileName = (String) entry.getValue();
            Path filePath = rootPath.resolve(fileName);

            if (!Files.exists(filePath)) {
                throw new IOException("Shard file not found: " + fileName);
            }

            tensorIndex.put(tensorName, filePath);
        }
    }

    private static void loadSingleFile(Path filePath, Map<String, Path> tensorIndex)
            throws IOException {
        Safetensors st = Safetensors.read(filePath);

        for (TensorEntry info : st.getTensors()) {
            tensorIndex.put(info.name(), filePath);
        }
    }
}
