package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.testkit.TestModels;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/**
 * A Boot app selects jinfer for embeddings and gets EmbeddingGemma 2 from properties alone: the
 * width the starter configures reaches the model's every call, and the modality list is the one the
 * attached projector really reads. Model-gated via {@link TestModels}.
 */
@Tag("integration")
class JinferEmbeddingAutoConfigurationIT {

    static final String MODEL = "hf.co/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";
    static final String MMPROJ = "hf.co/unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf";

    @Test
    void theConfiguredWidthReachesTheModel() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferEmbeddingAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.model.embedding=jinfer",
                        "spring.ai.jinfer.embedding.model=" + TestModels.require(MODEL),
                        "spring.ai.jinfer.embedding.dimensions=256")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            EmbeddingModel model = context.getBean(EmbeddingModel.class);
                            // the width a vector store sizes its table from
                            assertThat(model.dimensions()).isEqualTo(256);
                            // and the width every call without options embeds at
                            assertThat(model.embed("a sentence to embed")).hasSize(256);
                            assertThat(model.embed("another one")).hasSize(256);
                        });
    }

    @Test
    void theModelsOwnWidthIsTheDefault() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferEmbeddingAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.model.embedding=jinfer",
                        "spring.ai.jinfer.embedding.model=" + TestModels.require(MODEL))
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            EmbeddingModel model = context.getBean(EmbeddingModel.class);
                            assertThat(model.dimensions()).isEqualTo(768);
                            assertThat(model.embed("hello")).hasSize(768);
                        });
    }

    /**
     * The projector property is what makes media legal: the model takes them, text alone does not.
     */
    @Test
    void theProjectorPropertyAttachesTheTowers() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferEmbeddingAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.model.embedding=jinfer",
                        "spring.ai.jinfer.embedding.model=" + TestModels.require(MODEL),
                        "spring.ai.jinfer.embedding.companions.media=" + TestModels.require(MMPROJ))
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            com.qxotic.jinfer.spring.ai.JinferEmbeddingModel model =
                                    context.getBean(
                                            com.qxotic.jinfer.spring.ai.JinferEmbeddingModel.class);
                            // the towers are attached: the media content types are advertised
                            assertThat(model.dimensions()).isEqualTo(768);
                        });
    }
}
