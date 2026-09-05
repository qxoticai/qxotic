package com.qxotic.format.safetensors.impl;

import com.qxotic.format.json.Json;
import com.qxotic.format.safetensors.Safetensors;
import com.qxotic.format.safetensors.SafetensorsFormatException;
import com.qxotic.format.safetensors.SafetensorsIndex;
import com.qxotic.format.safetensors.TensorEntry;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

final class SafetensorsIndexImpl implements SafetensorsIndex {
    private static final String MODEL_SAFETENSORS = "model.safetensors";
    private static final String SAFETENSORS_INDEX = "model.safetensors.index.json";
    private static final String WEIGHT_MAP = "weight_map";
    private static final String INDEX_ROOT_ERROR = "Index JSON must be an object";
    private static final String WEIGHT_MAP_TYPE_ERROR =
            "'" + WEIGHT_MAP + "' keys and values must be strings in " + SAFETENSORS_INDEX;

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
            rootPath = resolveRootPath(path);
            loadSingleFile(path, tensorIndex);
        } else {
            throw new IOException("Path must be a directory or .safetensors file: " + path);
        }

        return new SafetensorsIndexImpl(rootPath, tensorIndex);
    }

    private static void loadSharded(Path rootPath, Path indexPath, Map<String, Path> tensorIndex)
            throws IOException {
        Map<String, String> weightMap;
        try {
            Map<?, ?> index =
                    requireObject(Json.parseMap(Files.readString(indexPath)), INDEX_ROOT_ERROR);
            weightMap = parseWeightMap(index.get(WEIGHT_MAP));
        } catch (Json.ParseException | SafetensorsFormatException e) {
            throw new SafetensorsFormatException(
                    "Invalid JSON in " + SAFETENSORS_INDEX + ": " + e.getMessage(), e);
        }

        for (Map.Entry<String, String> entry : weightMap.entrySet()) {
            String tensorName = entry.getKey();
            String fileName = entry.getValue();
            tensorIndex.put(tensorName, resolveShard(rootPath, fileName));
        }
    }

    private static Path resolveShard(Path rootPath, String fileName) throws IOException {
        final Path relative;
        try {
            relative = Path.of(fileName);
        } catch (InvalidPathException e) {
            throw new SafetensorsFormatException("Invalid shard path: " + fileName, e);
        }

        if (relative.isAbsolute()) {
            throw new SafetensorsFormatException("Shard path must be relative: " + fileName);
        }

        Path filePath = rootPath.resolve(relative).normalize();
        Path root = rootPath.toAbsolutePath().normalize();
        if (!filePath.toAbsolutePath().normalize().startsWith(root)) {
            throw new SafetensorsFormatException("Shard path escapes model directory: " + fileName);
        }
        if (!filePath.getFileName().toString().endsWith(".safetensors")) {
            throw new SafetensorsFormatException("Shard must be a .safetensors file: " + fileName);
        }
        if (!Files.isRegularFile(filePath)) {
            throw new IOException("Shard file not found: " + fileName);
        }
        return filePath;
    }

    private static void loadSingleFile(Path filePath, Map<String, Path> tensorIndex)
            throws IOException {
        Safetensors st = Safetensors.read(filePath);

        for (TensorEntry info : st.getTensors()) {
            tensorIndex.put(info.name(), filePath);
        }
    }

    private static Map<String, String> parseWeightMap(Object value) {
        Map<?, ?> raw =
                requireObject(value, "Missing '" + WEIGHT_MAP + "' in " + SAFETENSORS_INDEX);
        Map<String, String> weightMap = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            String key = requireString(entry.getKey(), WEIGHT_MAP_TYPE_ERROR);
            String file = requireString(entry.getValue(), WEIGHT_MAP_TYPE_ERROR);
            weightMap.put(key, file);
        }
        return weightMap;
    }

    private static Path resolveRootPath(Path safetensorsPath) {
        Path parent = safetensorsPath.getParent();
        return parent != null ? parent : safetensorsPath.toAbsolutePath().getParent();
    }

    private static Map<?, ?> requireObject(Object value, String message) {
        if (!(value instanceof Map)) {
            throw new SafetensorsFormatException(message);
        }
        return (Map<?, ?>) value;
    }

    private static String requireString(Object value, String message) {
        if (!(value instanceof String)) {
            throw new SafetensorsFormatException(message);
        }
        return (String) value;
    }
}
