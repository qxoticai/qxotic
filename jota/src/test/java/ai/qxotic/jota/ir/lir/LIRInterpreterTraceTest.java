package ai.qxotic.jota.ir.lir;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.TIRToLIRLowerer;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.LazyComputation;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import ai.qxotic.jota.testutil.TestKernels;
import org.junit.jupiter.api.Test;

class LIRInterpreterTraceTest {

    private static final int MANDEL_WIDTH = 320;
    private static final int MANDEL_HEIGHT = 240;
    private static final int MANDEL_ITER = 20;

    @Test
    void interpretsTracedGelu() {
        MemoryContext<MemorySegment> context = ContextFactory.ofMemorySegment();
        MemoryView<MemorySegment> input =
                MemoryHelpers.arange(context, DataType.FP32, 4).view(Shape.flat(4));
        Tensor inputTensor = Tensor.of(input);

        Tensor traced = Tracer.trace(inputTensor, Tensor::gelu);
        TIRGraph tirGraph = extractGraph(traced);

        LIRGraph lirGraph = new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(tirGraph));
        MemoryView<MemorySegment> output = allocateOutput(context, lirGraph.outputs().getFirst());

        new LIRInterpreter()
                .execute(lirGraph, List.of(input), List.of(), List.of(output), context);

        assertEquals(TestKernels.gelu(0.0f), readFloat(context, output, 0), 1e-4f);
        assertEquals(TestKernels.gelu(1.0f), readFloat(context, output, 1), 1e-4f);
        assertEquals(TestKernels.gelu(2.0f), readFloat(context, output, 2), 1e-4f);
        assertEquals(TestKernels.gelu(3.0f), readFloat(context, output, 3), 1e-4f);
    }

    @Test
    void interpretsTracedMandelbrot() {
        MemoryContext<MemorySegment> context = ContextFactory.ofMemorySegment();
        Tensor traced =
                Tracer.trace(
                        List.of(),
                        inputs ->
                                TestKernels.mandelbrotTensor(
                                        MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER));
        TIRGraph tirGraph = extractGraph(traced);

        LIRGraph lirGraph = new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(tirGraph));
        MemoryView<MemorySegment> output = allocateOutput(context, lirGraph.outputs().getFirst());

        new LIRInterpreter().execute(lirGraph, List.of(), List.of(), List.of(output), context);

        TestKernels.writeMandelbrotPpm(
                context,
                output,
                Path.of("target", "mandelbrot-lir.ppm"),
                MANDEL_WIDTH,
                MANDEL_HEIGHT,
                MANDEL_ITER);

        assertEquals(
                TestKernels.mandelbrotIter(0, 0, MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER),
                readFloat(context, output, 0),
                1e-3f);
        assertEquals(
                TestKernels.mandelbrotIter(
                        MANDEL_HEIGHT / 2, MANDEL_WIDTH / 2, MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER),
                readFloat(
                        context,
                        output,
                        (MANDEL_HEIGHT / 2L) * MANDEL_WIDTH + (MANDEL_WIDTH / 2L)),
                1e-3f);
        assertEquals(
                TestKernels.mandelbrotIter(
                        MANDEL_HEIGHT - 1, MANDEL_WIDTH - 1, MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER),
                readFloat(
                        context,
                        output,
                        (long) (MANDEL_HEIGHT - 1) * MANDEL_WIDTH + (MANDEL_WIDTH - 1)),
                1e-3f);
    }

    @Test
    void interpretsTracedPhoenix() {
        MemoryContext<MemorySegment> context = ContextFactory.ofMemorySegment();
        Tensor traced =
                Tracer.trace(
                        List.of(),
                        inputs ->
                                TestKernels.phoenixTensor(
                                        MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER));
        TIRGraph tirGraph = extractGraph(traced);

        LIRGraph lirGraph = new LIRStandardPipeline().run(new TIRToLIRLowerer().lower(tirGraph));
        MemoryView<MemorySegment> output = allocateOutput(context, lirGraph.outputs().getFirst());

        new LIRInterpreter().execute(lirGraph, List.of(), List.of(), List.of(output), context);

        TestKernels.writePhoenixPpm(
                context,
                output,
                Path.of("target", "phoenix-lir.ppm"),
                MANDEL_WIDTH,
                MANDEL_HEIGHT,
                MANDEL_ITER);

        assertEquals(
                TestKernels.phoenixIter(0, 0, MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER),
                readFloat(context, output, 0),
                1e-3f);
        assertEquals(
                TestKernels.phoenixIter(
                        MANDEL_HEIGHT / 2, MANDEL_WIDTH / 2, MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER),
                readFloat(
                        context,
                        output,
                        (MANDEL_HEIGHT / 2L) * MANDEL_WIDTH + (MANDEL_WIDTH / 2L)),
                1e-3f);
        assertEquals(
                TestKernels.phoenixIter(
                        MANDEL_HEIGHT - 1, MANDEL_WIDTH - 1, MANDEL_WIDTH, MANDEL_HEIGHT, MANDEL_ITER),
                readFloat(
                        context,
                        output,
                        (long) (MANDEL_HEIGHT - 1) * MANDEL_WIDTH + (MANDEL_WIDTH - 1)),
                1e-3f);
    }

    private TIRGraph extractGraph(Tensor traced) {
        LazyComputation computation = traced.computation().orElseThrow();
        Object graph = computation.attributes().get("graph");
        if (!(graph instanceof TIRGraph tirGraph)) {
            throw new IllegalStateException("Expected TIRGraph, got: " + graph);
        }
        return tirGraph;
    }

    private MemoryView<MemorySegment> allocateOutput(
            MemoryContext<MemorySegment> context, BufferRef buffer) {
        Memory<MemorySegment> memory =
                context.memoryAllocator().allocateMemory(buffer.dataType(), buffer.layout().shape());
        return MemoryView.of(memory, buffer.dataType(), buffer.layout());
    }

    private float readFloat(
            MemoryContext<MemorySegment> context, MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return access.readFloat(typedView.memory(), offset);
    }
}
