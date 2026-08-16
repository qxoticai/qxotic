package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.common.AbstractEmbeddingModelIT;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import dev.langchain4j.model.embedding.request.EmbeddingParameter;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j embedding compliance kit against {@link JinferEmbeddingModel} on Qwen3-Embedding
 * 0.6B: batch order, dimension, usage, query-vs-document divergence, the loud dimensions rejection,
 * and listener dispatch. Model-gated via @EnabledIf.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferEmbeddingModelTckIT#modelAvailable")
class JinferEmbeddingModelTckIT extends AbstractEmbeddingModelIT {

    private static final String REF =
            "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    // one shared model behind a non-AutoCloseable wrapper: JUnit closes AutoCloseable
    // parameterized-test arguments after EVERY invocation (see the chat TCK's models() note)
    private static JinferEmbeddingModel shared;

    // modelWith/failingModelWith products are built inside kit test bodies; nothing else closes
    private static final List<JinferEmbeddingModel> created =
            Collections.synchronizedList(new ArrayList<>());

    @AfterAll
    static void unload() {
        if (shared != null) shared.close();
        created.forEach(JinferEmbeddingModel::close);
        created.clear();
    }

    private static JinferEmbeddingModel build(EmbeddingModelListener listener) {
        return JinferEmbeddingModel.builder()
                .modelPath(TestModels.require(REF))
                .contextLength(1024)
                .listeners(listener == null ? List.of() : List.of(listener))
                .build();
    }

    @Override
    protected List<EmbeddingModel> models() {
        if (shared == null) shared = build(null);
        JinferEmbeddingModel m = shared;
        return List.of(
                new EmbeddingModel() {
                    @Override
                    public EmbeddingResponse embed(EmbeddingRequest request) {
                        return m.embed(request);
                    }

                    @Override
                    public EmbeddingResponse doEmbed(EmbeddingRequest request) {
                        return m.embed(request);
                    }

                    @Override
                    public int dimension() {
                        return m.dimension();
                    }

                    @Override
                    public String modelName() {
                        return m.modelName();
                    }

                    @Override
                    public Set<EmbeddingParameter<?>> supportedParameters() {
                        return m.supportedParameters();
                    }
                });
    }

    @Override
    protected EmbeddingModel modelWith(EmbeddingModelListener listener) {
        JinferEmbeddingModel m = build(listener);
        created.add(m);
        return m;
    }

    @Override
    protected EmbeddingModel failingModelWith(EmbeddingModelListener listener) {
        JinferEmbeddingModel m = build(listener);
        m.close(); // use-after-close: the honest call-time failure (a missing GGUF fails at
        // build time and could never be returned here)
        created.add(m);
        return m;
    }

    @Override
    protected boolean supportsDimensionsParameter() {
        return true; // Qwen3-Embedding is Matryoshka-trained (32..native)
    }

    @Override
    protected boolean supportsImageInput() {
        return false; // text-only embeddings; image input rejects loudly (core content-type gate)
    }

    @Override
    protected boolean supportsInterleavedInput() {
        return false;
    }
}
