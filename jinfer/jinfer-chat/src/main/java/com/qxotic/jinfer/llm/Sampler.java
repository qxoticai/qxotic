package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.Set;

/** Composable token sampling over a dense FP32 logits view. */
@FunctionalInterface
public interface Sampler {
    int sampleToken(MemoryView<?> logits);

    Sampler ARGMAX =
            logits -> {
                MemoryView<MemorySegment> values = checked(logits);
                return Ops.argmax(values, 0, size(values));
            };

    static Sampler banning(Sampler inner, Set<Integer> bannedTokens) {
        Objects.requireNonNull(inner, "inner");
        Objects.requireNonNull(bannedTokens, "bannedTokens");
        if (bannedTokens.isEmpty()) return inner;
        int[] banned = bannedTokens.stream().mapToInt(Integer::intValue).toArray();
        return logits -> {
            MemoryView<MemorySegment> values = checked(logits);
            int n = size(values);
            for (int token : banned) {
                if (token < 0 || token >= n)
                    throw new IllegalArgumentException("banned token " + token + " outside logits");
                set(values, token, Float.NEGATIVE_INFINITY);
            }
            return inner.sampleToken(values);
        };
    }

    private static MemoryView<MemorySegment> checked(MemoryView<?> logits) {
        Views.requireF32(logits, "logits");
        Views.requireContiguous(logits, "logits");
        MemoryView<MemorySegment> values = Views.castToSegmentBacked(logits, "logits");
        Views.checkAlive(values, "logits");
        if (values.shape().size() == 0) throw new IllegalArgumentException("empty logits");
        return values;
    }

    private static int size(MemoryView<?> values) {
        return Math.toIntExact(values.shape().size());
    }

    private static void set(MemoryView<MemorySegment> values, int index, float value) {
        values.memory()
                .base()
                .set(
                        ValueLayout.JAVA_FLOAT,
                        values.byteOffset() + (long) index * Float.BYTES,
                        value);
    }
}
