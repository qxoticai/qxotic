package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.common.AbstractAiServiceWithJsonSchemaIT;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j POJO-extraction battery ({@code AbstractAiServiceWithJsonSchemaIT}: primitives,
 * nested POJOs, enums, arrays/lists/sets of each, missing data, local dates, UUIDs...) against
 * JinferChatModel on LFM2.5-8B. AiServices reads {@code RESPONSE_FORMAT_JSON_SCHEMA} from the
 * provider and rides jinfer's grammar-constrained decoding: the schema is ENFORCED at the sampler,
 * so every extraction parses by construction - what the kit checks is that the right VALUES land in
 * the right fields end to end.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferAiServiceWithJsonSchemaTckIT#modelAvailable")
class JinferAiServiceWithJsonSchemaTckIT extends AbstractAiServiceWithJsonSchemaIT {

    static boolean modelAvailable() {
        return TestModels.find(JinferChatModelTckIT.REF).isPresent();
    }

    private static JinferChatModel model;

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    static JinferChatModel shared() {
        if (model == null) {
            model =
                    JinferChatModel.builder()
                            .modelPath(TestModels.require(JinferChatModelTckIT.REF))
                            .contextLength(8192)
                            .maxOutputTokens(512)
                            .temperature(0.0)
                            .thinking(JinferChatModelTckIT.tckThinking())
                            .seed(7L)
                            .build();
        }
        return model;
    }

    /**
     * jinfer enforces the response schema AT THE SAMPLER (grammar-constrained decoding), which is
     * the strict mode this hook describes - stricter, in fact, than the hosted providers it was
     * written for. The kit's own note: "LLMs in strict JSON schema mode return enums for some
     * reason, even if it is optional and no data available", so it drops the assertion that an
     * absent enum comes back null. Everything else in that test (absent strings, numbers, maps,
     * lists, arrays, nested POJOs, dates) is asserted as usual and passes.
     */
    @Override
    protected boolean isStrictJsonSchemaEnabled(ChatModel model) {
        return true;
    }

    @Override
    protected List<ChatModel> models() {
        return List.of(TckShield.chat(shared()));
    }
}
