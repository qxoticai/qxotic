package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;

/**
 * HuggingFace as a {@link ModelSource}. The no-arg constructor honors {@code HF_ENDPOINT}, the
 * variable every HuggingFace client reads; the {@link URI} constructor points at a mirror
 * explicitly, when the environment is not yours to set.
 */
public final class HuggingFaceSource implements ModelSource {

    private final RepositorySource inner;

    public HuggingFaceSource() {
        this(null);
    }

    public HuggingFaceSource(URI endpoint) {
        inner = new RepositorySource(ModelRef.Host.HF, endpoint);
    }

    @Override
    public boolean supports(ModelRef ref) {
        return inner.supports(ref);
    }

    @Override
    public List<RemoteFile> list(ModelRef ref, String dir) throws IOException {
        return inner.list(ref, dir);
    }

    @Override
    public void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException {
        inner.fetch(ref, file, into);
    }

    @Override
    public String toString() {
        return inner.toString();
    }
}
