package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferSpeechModel;
import com.qxotic.jinfer.spring.ai.JinferTranscriptionModel;
import com.qxotic.jinfer.testkit.TestModels;
import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.observation.tck.TestObservationRegistryAssert;
import java.nio.file.Path;
import java.util.Locale;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.core.io.ByteArrayResource;

/**
 * The auto-configured speech and transcription models, end to end: a phrase spoken, then heard, and
 * both calls observed through the application's registry. Model-gated via {@link TestModels}.
 */
@Tag("integration")
class JinferAudioAutoConfigurationIT {

    @Test
    void aSpokenPhraseIsHeardAndBothCallsAreObserved() {
        Path speech =
                TestModels.require(
                        "hf.co/remixerdec/Inflect-Nano-v2-GGUF/inflect_nano_v2_q8_0.gguf");
        Path transcription = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf");
        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                JinferSpeechAutoConfiguration.class,
                                JinferTranscriptionAutoConfiguration.class))
                .withBean(TestObservationRegistry.class, TestObservationRegistry::create)
                .withPropertyValues(
                        "spring.ai.jinfer.speech.model=" + speech,
                        "spring.ai.jinfer.transcription.model=" + transcription)
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            byte[] wav =
                                    context.getBean(JinferSpeechModel.class)
                                            .call("Hello from Paris.");
                            String heard =
                                    context.getBean(JinferTranscriptionModel.class)
                                            .transcribe(new ByteArrayResource(wav));
                            assertThat(heard.toLowerCase(Locale.ROOT)).contains("paris");
                            TestObservationRegistryAssert.assertThat(
                                            context.getBean(TestObservationRegistry.class))
                                    .hasNumberOfObservationsWithNameEqualTo(
                                            "gen_ai.client.operation", 2)
                                    .hasAnObservationWithAKeyValue(
                                            "gen_ai.operation.name", "text_to_speech")
                                    .hasAnObservationWithAKeyValue("gen_ai.system", "jinfer")
                                    // each names the GGUF that answered it
                                    .hasAnObservationWithAKeyValue(
                                            "gen_ai.request.model", speech.getFileName().toString())
                                    .hasAnObservationWithAKeyValue(
                                            "gen_ai.request.model",
                                            transcription.getFileName().toString());
                        });
    }
}
