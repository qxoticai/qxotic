package ai.qxotic.model.llm.granite;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.SpanLoader;
import ai.qxotic.model.llm.llama.BaseLlamaLoader;
import ai.qxotic.model.llm.llama.DefaultKernelOps;
import ai.qxotic.model.llm.llama.Llama;

public class GraniteLoader extends BaseLlamaLoader {

    static final String GRANITE_ARCH = "granite";

    public GraniteLoader(GGUF gguf) {
        super(gguf);
    }

    @Override
    public Granite loadModel(Llama.Configuration configuration) {
        return new Granite(
                configuration, DefaultKernelOps.getKernelOps(), DefaultKernelOps.getSpanFactory());
    }

    @Override
    public Llama.Configuration loadConfiguration(int maxTokens, SpanLoader spanLoader) {
        String arch = gguf.getValue(String.class, "general.architecture");
        if (!GRANITE_ARCH.equals(arch)) {
            throw new IllegalArgumentException(
                    "general.architecture expected " + GRANITE_ARCH + " but found " + arch);
        }
        return super.loadConfiguration(maxTokens, spanLoader)
                .with(
                        b ->
                                b.ropeIsNeoxStyle(false) // Granite uses classic RoPE
                                        .attentionScale(
                                                gguf.getValueOrDefault(
                                                        float.class,
                                                        arch + ".attention.scale",
                                                        Float.NaN))
                                        .residualScale(
                                                gguf.getValueOrDefault(
                                                        float.class,
                                                        arch + ".residual_scale",
                                                        Float.NaN))
                                        .logitScale(
                                                gguf.getValueOrDefault(
                                                        float.class,
                                                        arch + ".logit_scale",
                                                        Float.NaN))
                                        .embeddingScale(
                                                gguf.getValueOrDefault(
                                                        float.class,
                                                        arch + ".embedding_scale",
                                                        Float.NaN)));
    }
}
