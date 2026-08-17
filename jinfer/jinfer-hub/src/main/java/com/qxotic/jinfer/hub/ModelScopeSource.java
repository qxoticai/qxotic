package com.qxotic.jinfer.hub;

/** ModelScope as a {@link ModelSource}. Honors {@code MODELSCOPE_ENDPOINT}. */
public final class ModelScopeSource extends RepositorySource {

    public ModelScopeSource() {
        super(ModelRef.Host.MODELSCOPE, null);
    }
}
