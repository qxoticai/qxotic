package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.safetensors.Builder;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.SafetensorsIndex;
import java.io.IOException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Path;

public final class ImplAccessor {
    private ImplAccessor() {}

    public static Safetensors read(ReadableByteChannel channel) throws IOException {
        return ReaderImpl.read(channel);
    }

    public static SafetensorsIndex loadIndex(Path rootPath) throws IOException {
        return SafetensorsIndexImpl.load(rootPath);
    }

    public static Builder newBuilder() {
        return new BuilderImpl();
    }

    public static Builder newBuilder(Safetensors safetensors) {
        return BuilderImpl.fromExisting(safetensors);
    }

    public static int defaultAlignment() {
        return AlignmentSupport.DEFAULT_VALUE;
    }

    public static String alignmentKey() {
        return AlignmentSupport.KEY;
    }

    public static boolean isValidAlignment(int alignment) {
        return AlignmentSupport.isValid(alignment);
    }

    public static int parseAlignment(String alignment) {
        return AlignmentSupport.parse(alignment);
    }

    public static void write(Safetensors safetensors, WritableByteChannel byteChannel)
            throws IOException {
        WriterImpl.writeImpl(safetensors, byteChannel);
    }

    public static String toString(Safetensors safetensors, boolean showKeys, boolean showTensors) {
        return SafetensorsFormatter.toString(safetensors, showKeys, showTensors);
    }
}
