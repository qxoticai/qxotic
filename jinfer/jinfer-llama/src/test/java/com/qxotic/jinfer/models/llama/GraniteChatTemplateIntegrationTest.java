package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The real Granite GGUF templates render their own prompt and tool dialects. */
@Tag("integration")
final class GraniteChatTemplateIntegrationTest {

    private static final String GRANITE_41 =
            "hf.co/ibm-granite/granite-4.1-3b-GGUF/granite-4.1-3b-Q8_0.gguf";
    private static final String GRANITE_42 =
            "hf.co/ibm-granite/granite-4.2-3b-GGUF/granite-4.2-3b-Q8_0.gguf";

    private static final Map<String, Object> WEATHER =
            Map.of(
                    "type",
                    "function",
                    "function",
                    Map.of(
                            "name",
                            "get_weather",
                            "description",
                            "Get current weather",
                            "parameters",
                            Map.of(
                                    "type",
                                    "object",
                                    "properties",
                                    Map.of("city", Map.of("type", "string")),
                                    "required",
                                    List.of("city"))));

    @Test
    void granite41RendersRoleAndJsonToolScaffolding() throws Exception {
        Checkpoint checkpoint = Checkpoint.load(GRANITE_41);
        String prompt = checkpoint.render(false);

        assertTrue(prompt.contains("<|start_of_role|>system<|end_of_role|>"));
        assertTrue(prompt.contains("get_weather"));
        assertTrue(prompt.contains("<tool_call>"));
        assertTrue(prompt.contains("<|start_of_role|>user<|end_of_role|>Weather?"));
        assertTrue(prompt.endsWith("<|start_of_role|>assistant<|end_of_role|>"));
        int startRole = SpecialTokens.require(checkpoint.tokenizer(), "<|start_of_role|>");
        assertTrue(checkpoint.encoded(false).stream().anyMatch(token -> token == startRole));
    }

    @Test
    void granite42RendersChatMlAndFunctionToolScaffolding() throws Exception {
        Checkpoint checkpoint = Checkpoint.load(GRANITE_42);
        String prompt = checkpoint.render(false);

        assertTrue(prompt.contains("<|im_start|>system"));
        assertTrue(prompt.contains("get_weather"));
        assertTrue(prompt.contains("<function="));
        assertTrue(prompt.contains("<|im_start|>user\nWeather?<|im_end|>"));
        assertTrue(prompt.endsWith("<|im_start|>assistant\n<think></think>"));
        int imStart = SpecialTokens.require(checkpoint.tokenizer(), "<|im_start|>");
        assertTrue(checkpoint.encoded(false).stream().anyMatch(token -> token == imStart));
    }

    private record Checkpoint(Tokenizer tokenizer, String template) {

        static Checkpoint load(String reference) throws Exception {
            Path path = TestModels.require(reference);
            try (FileChannel file = FileChannel.open(path)) {
                GGUF gguf = ModelLoader.readGguf(file, path.toString());
                return new Checkpoint(
                        GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf),
                        gguf.getString("tokenizer.chat_template"));
            }
        }

        String render(boolean thinking) {
            Map<String, Object> variables = new LinkedHashMap<>();
            variables.put("messages", List.of(Map.of("role", "user", "content", "Weather?")));
            variables.put("tools", List.of(WEATHER));
            variables.put("add_generation_prompt", true);
            variables.put("enable_thinking", thinking);
            variables.put("bos_token", special(SpecialTokens.bos(tokenizer)));
            variables.put("eos_token", special(SpecialTokens.eos(tokenizer)));
            return JinjaRenderer.template(template).render(variables);
        }

        IntSequence encoded(boolean thinking) {
            return SpecialTokens.encode(tokenizer, render(thinking));
        }

        private String special(java.util.OptionalInt id) {
            return id.isPresent() ? tokenizer.vocabulary().token(id.getAsInt()) : "";
        }
    }
}
