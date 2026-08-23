///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --release 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-gemma4
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
//SOURCES scripts/Models.java

// Gemma 4 vision through Jinfer's multimodal chat API.
// The same code runs on every Gemma 4 size. Only the model and media references change:
//
//   E2B:  jbang GemmaVision.java cat.jpg "What is in this image?"
//   E4B:  jbang GemmaVision.java cat.jpg "Describe it" \
//             hf.co/unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
//             hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf
//   12B:  jbang GemmaVision.java chart.png "Read the values off this chart" \
//             hf.co/unsloth/gemma-4-12b-it-GGUF:Q8_0 \
//             hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf
//
// The media companion contains the vision encoder and projector. Adjust the image token budget to
// trade detail for speed:
//     -Djinfer.gemma4.imageTokenBudget=70|140|280|560|1120

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;

import java.nio.file.Path;

public class GemmaVision {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println(
                    "usage: GemmaVision <image> <prompt> [model-ref] [mmproj-ref]");
            System.exit(2);
        }
        Path image = Path.of(args[0]);
        String prompt = args[1];
        String modelRef = Models.gemma(args, 2);
        String mediaRef = Models.gemmaMmproj(args, 3);

        try (var model = JinferChatModel.builder()
                .model(modelRef)
                .companion("media", mediaRef)
                .contextLength(4096)
                .maxOutputTokens(300)
                .thinking(false)
                .build()) {
            var message = UserMessage.from(
                    TextContent.from(prompt), ImageContent.from(image.toUri()));
            System.err.printf("image: %s%nmodel: %s%n", image, modelRef);
            System.out.println("\n=== Gemma 4 says ===");
            System.out.println(model.chat(message).aiMessage().text());
        }
    }
}
