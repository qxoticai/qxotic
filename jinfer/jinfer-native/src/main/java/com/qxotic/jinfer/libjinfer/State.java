package com.qxotic.jinfer.libjinfer;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContextModel;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.EmbeddingModel;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * The raw door: one state of your own over a model, fed tokens, read for logits or an embedding. It
 * holds up to {@code contextCapacity} tokens of context.
 */
@SuppressWarnings({"rawtypes", "unchecked"})
final class State {
    private final Model model;
    private final ContextModel cm; // erased: the state's type is the model's business
    private final ContextState state;

    State(Model model, int contextCapacity) {
        this.model = model;
        ContextModel<?, ?, ?> typed = model.contextModel();
        this.cm = typed;
        int max = typed.configuration().maxContextLength();
        int capacity = contextCapacity <= 0 ? model.capacity : Math.min(contextCapacity, max);
        this.state = cm.newState(capacity);
        model.retain();
    }

    int position() {
        return state.position();
    }

    void reset() {
        state.reset();
    }

    int ingest(int[] tokens) {
        if (state.position() + tokens.length > state.contextCapacity()) {
            throw Errors.invalid(
                    "ingesting "
                            + tokens.length
                            + " tokens at position "
                            + state.position()
                            + " exceeds the capacity of "
                            + state.contextCapacity());
        }
        if (model.kind == Model.Kind.EMBEDDING) {
            // a retrieval checkpoint attends bidirectionally: the sequence goes in as one packed
            // batch
            if (tokens.length > state.batchCapacity()) {
                throw Errors.invalid(
                        "an embedding model ingests at most "
                                + state.batchCapacity()
                                + " tokens at once, got "
                                + tokens.length);
            }
            cm.ingest(state, Batch.pack(new int[][] {tokens}));
            return state.position();
        }
        int batch = state.batchCapacity();
        for (int at = 0; at < tokens.length; at += batch) {
            cm.ingest(
                    state,
                    Batch.prefill(
                            Arrays.copyOfRange(tokens, at, Math.min(tokens.length, at + batch))));
        }
        return state.position();
    }

    float[] logits() {
        if (!(cm instanceof LanguageModel lm))
            throw Errors.unsupported(model.name + " projects no logits");
        if (state.position() == 0) throw Errors.invalid("nothing ingested yet");
        return floats(lm.logits(state), "logits");
    }

    float[] embedding() {
        if (!(cm instanceof EmbeddingModel em))
            throw Errors.unsupported(model.name + " projects no embedding");
        if (state.position() == 0) throw Errors.invalid("nothing ingested yet");
        float[][] out = new float[1][];
        em.projectEmbedding(state, view -> out[0] = floats((MemoryView<?>) view, "embedding"));
        return out[0];
    }

    private static float[] floats(MemoryView<?> view, String what) {
        MemoryView<MemorySegment> source = Views.castToSegmentBacked(view, what);
        Views.requireF32(source, what);
        Views.requireContiguous(source, what);
        int n = (int) source.shape().size();
        float[] out = new float[n];
        Views.copyToArray(source, 0, out, 0, n, what);
        return out;
    }

    void close() {
        state.close();
        model.release();
    }
}
