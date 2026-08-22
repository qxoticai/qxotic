package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/**
 * The scalar kernels must produce the SAME TOKENS as the vector ones. A fallback that runs but
 * computes something else is worse than no fallback, and {@code -Djinfer.vectorBitSize=0} is only
 * useful as a debugging and A/B tool if it is answer-preserving.
 *
 * <p>Float sums differ - the vector path accumulates in a different order, so logits diverge in the
 * third decimal. That is expected and harmless; what must not change is which token wins. This
 * asserts the whole greedy walk, so a single flipped argmax anywhere fails it.
 *
 * <p>Runs each mode in a FORKED JVM: {@code VECTOR_BIT_SIZE} is a {@code static final} read once
 * per process, so one JVM cannot exercise both.
 */
final class VectorScalarParityTest {

    private static final String PROMPT = "The capital of France is";
    private static final int STEPS = 16;

    @Test
    void scalarKernelsAgreeWithVectorKernelsTokenForToken() throws Exception {
        Path gguf =
                TestModels.require(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
        String vector = walk(gguf, null);
        String scalar = walk(gguf, "-Djinfer.vectorBitSize=0");

        assertTrue(vector.split(" ").length == STEPS, "vector run produced: " + vector);
        assertEquals(
                vector,
                scalar,
                "scalar kernels diverged from the vector kernels - a fallback that computes"
                        + " something else is worse than no fallback");
    }

    /** Runs {@link #main} in a fresh JVM and returns the greedy token walk it printed. */
    private static String walk(Path gguf, String extraFlag) throws Exception {
        List<String> cmd = new ArrayList<>();
        cmd.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        cmd.add("--add-modules");
        cmd.add("jdk.incubator.vector");
        cmd.add("--enable-native-access=ALL-UNNAMED");
        if (extraFlag != null) cmd.add(extraFlag);
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(VectorScalarParityTest.class.getName());
        cmd.add(gguf.toString());

        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String out;
        try (var in = p.getInputStream()) {
            out = new String(in.readAllBytes());
        }
        assertTrue(p.waitFor(10, TimeUnit.MINUTES), "forked JVM timed out");
        assertEquals(0, p.exitValue(), "forked JVM failed:\n" + out);
        return out.lines()
                .filter(l -> l.startsWith("WALK "))
                .map(l -> l.substring(5).trim())
                .findFirst()
                .orElseThrow(() -> new AssertionError("no WALK line in:\n" + out));
    }

    /** Forked entry point: greedy-decode {@link #STEPS} tokens and print them. */
    public static void main(String[] args) throws Exception {
        try (Arena weights = Arenas.newCrossThread()) {
            System.out.println("WALK " + greedy(Models.load(Path.of(args[0]), weights)));
        }
    }

    private static <S extends com.qxotic.jinfer.ContextState> String greedy(LoadedModel<S> loaded) {
        var model = loaded.model();
        int vocab = model.configuration().vocabularySize();
        S state = model.newState(512, 512);
        try {
            for (Batch b :
                    Batch.prepare(
                            List.of(Batch.prefill(loaded.tokenizer().encodeToArray(PROMPT))), 512))
                model.ingest(state, b);
            StringBuilder walk = new StringBuilder();
            int token = argmax(model.logits(state), vocab);
            for (int i = 0; i < STEPS; i++) {
                walk.append(token).append(' ');
                model.ingest(state, Batch.step(token));
                token = argmax(model.logits(state), vocab);
            }
            return walk.toString().trim();
        } finally {
            ((RuntimeState) state).close();
        }
    }

    private static int argmax(MemoryView<?> view, int n) {
        MemoryView<MemorySegment> v = Views.castToSegmentBacked(view, "logits");
        float[] logits =
                v.memory()
                        .base()
                        .asSlice(v.byteOffset(), (long) n * Float.BYTES)
                        .toArray(ValueLayout.JAVA_FLOAT);
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
}
