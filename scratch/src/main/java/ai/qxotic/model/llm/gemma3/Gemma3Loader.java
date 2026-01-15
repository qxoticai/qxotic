package ai.qxotic.model.llm.gemma3;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.SpanLoader;
import ai.qxotic.model.llm.llama.BaseLlamaLoader;
import ai.qxotic.model.llm.llama.DefaultKernelOps;
import ai.qxotic.model.llm.llama.Llama;
import ai.qxotic.span.FloatUnaryOperator;


public class Gemma3Loader extends BaseLlamaLoader {

    static final String GEMMA3_ARCH = "gemma3";

    public Gemma3Loader(GGUF gguf) {
        super(gguf);
    }

    @Override
    public Gemma3 loadModel(Llama.Configuration configuration) {
        return new Gemma3(configuration, DefaultKernelOps.getKernelOps(), DefaultKernelOps.getSpanFactory());
    }

    @Override
    public Gemma3.Configuration loadConfiguration(int maxTokens, SpanLoader spanLoader) {
        String arch = gguf.getValue(String.class, "general.architecture");
        if (!GEMMA3_ARCH.equals(arch)) {
            throw new IllegalArgumentException("general.architecture expected " + GEMMA3_ARCH + " but found " + arch);
        }
        return super.loadConfiguration(maxTokens, spanLoader)
                .with(b -> b.ropeIsNeoxStyle(true))
                .with(b -> b.activationFunction(FloatUnaryOperator.GELU));
    }
}
