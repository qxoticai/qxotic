package com.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.TIRToLIRLowerer;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.*;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.TensorTracing;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.TestKernels;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class LIRInterpreterTraceTest {

    private static final int MANDEL_WIDTH = 320;
    private static final int MANDEL_HEIGHT = 240;
    private static final int MANDEL_ITER = 32;

    @Test
    void interpretsTracedGelu() {
        MemoryDomain<MemorySegment> domain = Environment.nativeMemoryDomain();
        MemoryView<MemorySegment> input =
                MemoryHelpers.arange(domain, DataType.FP32, 4).view(Shape.flat(4));
        Tensor inputTensor = Tensor.of(input);

        Tensor traced = Tracer.trace(inputTensor, Tensor::gelu);
        TIRGraph tirGraph = extractGraph(traced);

        LIRGraph lirGraph = new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(tirGraph));
        MemoryView<MemorySegment> output = allocateOutput(domain, lirGraph.outputs().getFirst());

        new LIRInterpreter().execute(lirGraph, List.of(input), List.of(), List.of(output), domain);

        assertEquals(TestKernels.gelu(0.0f), readFloat(domain, output, 0), 1e-4f);
        assertEquals(TestKernels.gelu(1.0f), readFloat(domain, output, 1), 1e-4f);
        assertEquals(TestKernels.gelu(2.0f), readFloat(domain, output, 2), 1e-4f);
        assertEquals(TestKernels.gelu(3.0f), readFloat(domain, output, 3), 1e-4f);
    }

    @Test
    void interpretsTracedMandelbrot() {
        MemoryDomain<MemorySegment> domain = Environment.nativeMemoryDomain();
        Tensor traced =
                Tracer.trace(
                        List.of(),
                        inputs ->
                                TestKernels.mandelbrotTensor(
                                        MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER));
        TIRGraph tirGraph = extractGraph(traced);

        LIRGraph lirGraph = new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(tirGraph));
        MemoryView<MemorySegment> output = allocateOutput(domain, lirGraph.outputs().getFirst());

        new LIRInterpreter().execute(lirGraph, List.of(), List.of(), List.of(output), domain);

        TestKernels.writeMandelbrotPpm(
                output,
                Path.of("target", "mandelbrot-lir.ppm"),
                MANDEL_WIDTH,
                MANDEL_HEIGHT,
                MANDEL_ITER);

        assertEquals(
                TestKernels.mandelbrotIter(0, 0, MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER),
                readFloat(domain, output, 0),
                1e-3f);
        assertEquals(
                TestKernels.mandelbrotIter(
                        MANDEL_HEIGHT / 2,
                        MANDEL_WIDTH / 2,
                        MANDEL_WIDTH,
                        MANDEL_HEIGHT,
                        MANDEL_ITER),
                readFloat(
                        domain, output, (MANDEL_HEIGHT / 2L) * MANDEL_WIDTH + (MANDEL_WIDTH / 2L)),
                1e-3f);
        assertEquals(
                TestKernels.mandelbrotIter(
                        MANDEL_HEIGHT - 1,
                        MANDEL_WIDTH - 1,
                        MANDEL_WIDTH,
                        MANDEL_HEIGHT,
                        MANDEL_ITER),
                readFloat(
                        domain,
                        output,
                        (long) (MANDEL_HEIGHT - 1) * MANDEL_WIDTH + (MANDEL_WIDTH - 1)),
                1e-3f);
    }

    private TIRGraph extractGraph(Tensor traced) {
        return TensorTracing.tracedGraph(traced)
                .orElseThrow(() -> new IllegalStateException("Expected traced IR graph"));
    }

    private MemoryView<MemorySegment> allocateOutput(
            MemoryDomain<MemorySegment> domain, BufferRef buffer) {
        Memory<MemorySegment> memory =
                domain.memoryAllocator().allocateMemory(buffer.dataType(), buffer.layout().shape());
        return MemoryView.of(memory, buffer.dataType(), buffer.layout());
    }

    private float readFloat(
            MemoryDomain<MemorySegment> domain, MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = domain.directAccess();
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return access.readFloat(typedView.memory(), offset);
    }
}
