package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeState;
import java.util.function.Function;

/**
 * Resume-state codec for uniform full-attention models (Llama, Granite): every layer stores
 * absolute-position K/V rows and there is no residue - a block is just the span's rows. Windowed
 * models pass ring widths; models with small recurrent state add a {@link
 * AbstractStateCodec#residue} trailer.
 */
public final class DenseStateCodec<S extends RuntimeState> extends AbstractStateCodec<S> {

    public DenseStateCodec(
            int layers,
            long kvDim,
            Function<S, FloatTensor[]> keys,
            Function<S, FloatTensor[]> values) {
        super(layers, l -> true, l -> kvDim, l -> 0, keys, values, 0L);
    }
}
