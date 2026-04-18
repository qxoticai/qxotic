package com.qxotic.format.safetensors;

import com.qxotic.format.safetensors.impl.ImplAccessor;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Reads and writes Safetensors headers (metadata + tensor descriptors). Works on headers only;
 * tensor payload bytes are not accessed.
 *
 * @see <a href="https://github.com/huggingface/safetensors">Safetensors specification</a>
 */
public interface Safetensors {

    /** Byte offset where tensor payload begins ({@code 8 + headerSize}). */
    long getTensorDataOffset();

    /** Tensor alignment in bytes (defaults to 1 when __alignment__ is absent). */
    int getAlignment();

    /** Metadata from the __metadata__ key (unmodifiable, empty if absent). */
    Map<String, String> getMetadata();

    /** All tensors in this file (unmodifiable, order preserved). */
    Collection<TensorEntry> getTensors();

    /** Returns tensor information, or null if not found. */
    TensorEntry getTensor(String tensorName);

    /** Returns tensor information as an Optional. */
    default Optional<TensorEntry> findTensor(String tensorName) {
        Objects.requireNonNull(tensorName, "tensorName");
        return Optional.ofNullable(getTensor(tensorName));
    }

    /** Returns tensor information, or throws if not found. */
    default TensorEntry requireTensor(String tensorName) {
        Objects.requireNonNull(tensorName, "tensorName");
        return findTensor(tensorName)
                .orElseThrow(() -> new IllegalArgumentException("tensor not found: " + tensorName));
    }

    default boolean containsTensor(String tensorName) {
        return getTensor(tensorName) != null;
    }

    /**
     * Absolute byte offset where tensor data begins ({@code getTensorDataOffset() +
     * tensor.byteOffset()}).
     */
    default long absoluteOffset(TensorEntry tensor) {
        Objects.requireNonNull(tensor, "tensor");
        return getTensorDataOffset() + tensor.byteOffset();
    }

    /** Reads safetensors metadata from a channel (header only, not tensor payload). */
    static Safetensors read(ReadableByteChannel channel) throws IOException {
        Objects.requireNonNull(channel, "channel");
        return ImplAccessor.read(channel);
    }

    /** Reads safetensors metadata from a file. */
    static Safetensors read(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        try (ReadableByteChannel channel =
                Channels.newChannel(
                        new BufferedInputStream(
                                Files.newInputStream(path, StandardOpenOption.READ), 1 << 16))) {
            return read(channel);
        }
    }

    /** Writes a Safetensors header to a channel. */
    static void write(Safetensors safetensors, WritableByteChannel byteChannel) throws IOException {
        Objects.requireNonNull(safetensors, "safetensors");
        Objects.requireNonNull(byteChannel, "byteChannel");
        ImplAccessor.write(safetensors, byteChannel);
    }

    /** Writes a Safetensors header to a file. */
    static void write(Safetensors safetensors, Path modelPath) throws IOException {
        Objects.requireNonNull(safetensors, "safetensors");
        Objects.requireNonNull(modelPath, "modelPath");
        try (WritableByteChannel byteChannel =
                Files.newByteChannel(
                        modelPath, StandardOpenOption.WRITE, StandardOpenOption.CREATE_NEW)) {
            write(safetensors, byteChannel);
        }
    }
}
