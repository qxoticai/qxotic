///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
// jinfer is a local (unpublished) build - install it to your ~/.m2 once, then jbang resolves it:
//     cd jinfer && mvn -q -DskipTests install
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES scripts/Models.java

// Gemma 4 vision (image -> text) through jinfer's multimodal chat API.
// Same code runs on every Gemma 4 size - only the model + mmproj references change:
//
//   E2B:  jbang GemmaVision.java cat.jpg "What is in this image?"
//   E4B:  jbang GemmaVision.java cat.jpg "Describe it" \
//             ~/models/unsloth/gemma-4-E4B-it-Q8_0.gguf \
//             ~/models/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf   (E-variants share the projector)
//   12B:  jbang GemmaVision.java chart.png "Read the values off this chart" \
//             ~/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf \
//             ~/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf
//
// The mmproj (SigLIP vision tower + projector) is loaded alongside the text GGUF; jinfer runs the
// image through it into ~256 soft tokens (Gemma's default 280 budget) and splices them between
// <|image>...<image|> in the prompt. Trade image detail for speed with the token budget:
//     -Djinfer.gemma4.imageTokenBudget=70|140|280|560|1120   (higher = more detail, more compute)
// No native jam lib needed - it falls back to the Java Vector backend automatically (pass
// -Djam.native.library.path=/path/to/libjam.so for full speed).

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
