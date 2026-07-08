package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeState;

import java.util.function.Function;

/** Resume-state codec for uniform full-attention models (Llama, Granite): every layer stores
 *  absolute-position K/V rows and there is no checkpoint — so a block is just the span's rows and
 *  every block is a resume point. Hybrid models (windows, recurrent checkpoints) extend
 *  {@link AbstractStateCodec} and add their own {@link AbstractStateCodec#checkpoint}. */
public final class DenseStateCodec<S extends RuntimeState> extends AbstractStateCodec<S> {

    public DenseStateCodec(int layers, long kvDim, Function<S, FloatTensor[]> keys, Function<S, FloatTensor[]> values) {
        super(layers, l -> true, l -> kvDim, keys, values, l -> 0L);
    }
}
