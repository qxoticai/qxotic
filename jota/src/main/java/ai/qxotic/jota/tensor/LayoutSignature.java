package ai.qxotic.jota.tensor;

import java.util.List;

record LayoutSignature(List<ValueSpec> inputs, ValueSpec output) {}
