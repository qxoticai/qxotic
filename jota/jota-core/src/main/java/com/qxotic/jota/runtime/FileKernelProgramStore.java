package com.qxotic.jota.runtime;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;

public final class FileKernelProgramStore implements KernelProgramStore {

    private static final String META_FILE = "program.meta";
    private static final String SOURCE_FILE = "program.src";
    private static final String BINARY_FILE = "program.bin";

    private final Path root;

    public FileKernelProgramStore(Path root) {
        this.root = Objects.requireNonNull(root, "root");
    }

    @Override
    public Path root() {
        return root;
    }

    @Override
    public void store(KernelProgram program, KernelCacheKey key) {
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(key, "key");
        Path dir = root.resolve(key.value());
        try {
            Files.createDirectories(dir);
            Properties props = new Properties();
            props.setProperty("kind", program.kind().name());
            props.setProperty("language", program.language());
            props.setProperty("entryPoint", program.entryPoint());
            for (Map.Entry<String, String> entry : program.options().entrySet()) {
                props.setProperty("option." + entry.getKey(), entry.getValue());
            }
            Path metaPath = dir.resolve(META_FILE);
            try (OutputStream out = Files.newOutputStream(metaPath)) {
                props.store(out, "kernel program metadata");
            }
            if (program.kind() == KernelProgram.Kind.SOURCE) {
                String source = requireSource(program.payload());
                Files.writeString(dir.resolve(SOURCE_FILE), source, StandardCharsets.UTF_8);
            } else {
                byte[] binary = requireBinary(program.payload());
                Files.write(dir.resolve(BINARY_FILE), binary);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to store kernel program " + key, e);
        }
    }

    @Override
    public Optional<KernelProgram> load(KernelCacheKey key) {
        Objects.requireNonNull(key, "key");
        Path dir = root.resolve(key.value());
        Path metaPath = dir.resolve(META_FILE);
        if (!Files.exists(metaPath)) {
            return Optional.empty();
        }
        Properties props = new Properties();
        try (InputStream in = Files.newInputStream(metaPath)) {
            props.load(in);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read kernel program metadata " + key, e);
        }
        KernelProgram.Kind kind = KernelProgram.Kind.valueOf(props.getProperty("kind"));
        String language = props.getProperty("language");
        String entryPoint = props.getProperty("entryPoint");
        Map<String, String> options = readOptions(props);
        Object payload;
        try {
            if (kind == KernelProgram.Kind.SOURCE) {
                payload = Files.readString(dir.resolve(SOURCE_FILE), StandardCharsets.UTF_8);
            } else {
                payload = Files.readAllBytes(dir.resolve(BINARY_FILE));
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read kernel program payload " + key, e);
        }
        return Optional.of(new KernelProgram(kind, language, payload, entryPoint, options));
    }

    private static String requireSource(Object payload) {
        if (payload instanceof String source) {
            return source;
        }
        throw new IllegalArgumentException("Expected source payload as String");
    }

    private static byte[] requireBinary(Object payload) {
        if (payload instanceof byte[] bytes) {
            return bytes;
        }
        throw new IllegalArgumentException("Expected binary payload as byte[]");
    }

    private static Map<String, String> readOptions(Properties props) {
        Map<String, String> options = new LinkedHashMap<>();
        for (String name : props.stringPropertyNames()) {
            if (name.startsWith("option.")) {
                options.put(name.substring("option.".length()), props.getProperty(name));
            }
        }
        return options;
    }
}
