package com.qxotic.jinfer.hub;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/** ModelScope as a {@link ModelSource}. Honors {@code MODELSCOPE_ENDPOINT}. */
public final class ModelScopeSource implements ModelSource {

    private final RepositorySource inner = new RepositorySource(ModelRef.Host.MODELSCOPE, null);

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
