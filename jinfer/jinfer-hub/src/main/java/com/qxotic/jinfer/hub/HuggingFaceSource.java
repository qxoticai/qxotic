package com.qxotic.jinfer.hub;

import java.net.URI;

/**
 * HuggingFace as a {@link ModelSource}. The no-arg constructor honors {@code HF_ENDPOINT}, the
 * variable every HuggingFace client reads; the {@link URI} constructor points at a mirror
 * explicitly, when the environment is not yours to set.
 */
public final class HuggingFaceSource extends RepositorySource {

    public HuggingFaceSource() {
        this(null);
    }

    public HuggingFaceSource(URI endpoint) {
        super(ModelRef.Host.HF, endpoint);
    }
}
