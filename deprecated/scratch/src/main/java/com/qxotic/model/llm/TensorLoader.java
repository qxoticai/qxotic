package com.qxotic.model.llm;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.model.llm.llama.BF16Span;
import com.qxotic.model.llm.llama.F32BBSpan;
import com.qxotic.model.llm.llama.F32Span;
import com.qxotic.model.llm.llama.Q4_0BBSpan;
import com.qxotic.model.llm.llama.Q4_0Span;
import com.qxotic.model.llm.llama.Q4_1Span;
import com.qxotic.model.llm.llama.Q8_0BBSpan;
import com.qxotic.model.llm.llama.Q8_0Span;
import com.qxotic.model.llm.llama.Util;
import com.qxotic.span.FloatSpan;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class TensorLoader {
    private static SpanLoader loaderFromTensorDataBB(
            long tensorDataOffset, FileChannel fileChannel) {
        return baseTensor -> {
            TensorEntry tensorEntry = (TensorEntry) baseTensor;
            GGMLType ggmlType = tensorEntry.ggmlType();
            long sizeInBytes = ggmlType.byteSizeForShape(tensorEntry.shape());
            ByteBuffer byteBuffer = null;
            try {
                byteBuffer =
                        fileChannel
                                .map(
                                        FileChannel.MapMode.READ_ONLY,
                                        tensorDataOffset + tensorEntry.offset(),
                                        sizeInBytes)
                                .order(ByteOrder.nativeOrder());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            FloatSpan span =
                    switch (ggmlType) {
                        case F32 -> new F32BBSpan(byteBuffer);
                        case Q4_0 -> new Q4_0BBSpan(byteBuffer);
                        case Q8_0 -> new Q8_0BBSpan(byteBuffer);
                        default ->
                                throw new UnsupportedOperationException(
                                        "Quantization not supported " + ggmlType);
                    };
            assert span.size() == Util.numberOfElements(tensorEntry.shape());
            return span;
        };
    }

    private static SpanLoader loaderFromTensorData(MemorySegment tensorData) {
        return baseTensor -> {
            TensorEntry tensorEntry = (TensorEntry) baseTensor;
            long numberOfElements = Util.numberOfElements(tensorEntry.shape());
            GGMLType ggmlType = tensorEntry.ggmlType();
            long sizeInBytes = ggmlType.byteSizeFor(numberOfElements);
            MemorySegment memorySegment = tensorData.asSlice(tensorEntry.offset(), sizeInBytes);
            FloatSpan span =
                    switch (ggmlType) {
                        case F32 -> new F32Span(memorySegment);
                        case Q4_0 -> new Q4_0Span(memorySegment);
                        case Q8_0 -> new Q8_0Span(memorySegment);
                        default ->
                                throw new UnsupportedOperationException(
                                        "Quantization not supported " + ggmlType);
                    };
            assert span.size() == Util.numberOfElements(tensorEntry.shape());
            return span;
        };
    }

    public static SpanLoader loaderFromTensorDataMM(
            long tensorDataOffset, FileChannel fileChannel) {
        return baseTensor -> {
            TensorEntry tensorEntry = (TensorEntry) baseTensor;
            GGMLType ggmlType = tensorEntry.ggmlType();
            long sizeInBytes = ggmlType.byteSizeForShape(tensorEntry.shape());
            MemorySegment memorySegment = null;
            try {
                memorySegment =
                        fileChannel.map(
                                FileChannel.MapMode.READ_ONLY,
                                tensorDataOffset + tensorEntry.offset(),
                                sizeInBytes,
                                Arena.ofAuto());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            FloatSpan span =
                    switch (ggmlType) {
                        case F32 -> new F32Span(memorySegment);
                        case Q4_0 -> new Q4_0Span(memorySegment);
                        case Q8_0 -> new Q8_0Span(memorySegment);
                        case Q4_1 -> new Q4_1Span(memorySegment);
                        case BF16 -> new BF16Span(memorySegment);
                        default ->
                                throw new UnsupportedOperationException(
                                        "Quantization not supported " + ggmlType);
                    };
            assert span.size() == Util.numberOfElements(tensorEntry.shape());
            return span;
        };
    }
}
