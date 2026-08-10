package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.RuntimeState;
import java.util.Optional;

/**
 * Prompt-cache capability: a model that can externalize its resume-state carries a {@link
 * StateCodec}. Lives beside the cache (jinfer-kernels), not on {@code LanguageModel}: the codec
 * deals in raw state memory, which is a kernels concern, and the core model API stays free of it.
 * Consumers test {@code instanceof Cacheable} - absence IS the "no caching" signal (no sentinel).
 */
public interface Cacheable<S extends RuntimeState> {

    /** The resume-state codec for this model. Stateless. */
    Optional<StateCodec<S>> stateCodec();
}
