///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --release 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-langchain4j:0.1.0
//SOURCES scripts/Models.java

// Gemma 4 with MULTIPLE images in one prompt (per the docs: several image blocks per turn).
// Each image becomes its own <|image>...<image|> soft-token span; the model reasons across all of them.
//
//   Install once:  cd jinfer && mvn -q -DskipTests install
//   Run (12B is worth it for cross-image reasoning):
//     jbang GemmaVisionMulti.java "Which image has more animals, and by how many?" a.jpg b.jpg \
//         -- \
//         ~/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf \
//         ~/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf
//   Defaults to E2B if no model references are given.

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class GemmaVisionMulti {

    public static void main(String[] args) {
        int separator = List.of(args).indexOf("--");
        if (args.length < 3 || (separator >= 0 && (separator < 3 || args.length != separator + 3))) {
            System.err.println(
                    "usage: GemmaVisionMulti <prompt> <image1> <image2> [image3 ...]"
                            + " [-- <model-ref> <mmproj-ref>]");
            System.exit(2);
        }
        String prompt = args[0];

        int end = separator < 0 ? args.length : separator;
        String modelRef = separator < 0 ? Models.gemma(args, end) : args[separator + 1];
        String mediaRef = separator < 0 ? Models.gemmaMmproj(args, end) : args[separator + 2];

        // Build one user turn whose content interleaves the prompt text with N image parts.
        List<Content> content = new ArrayList<>();
        content.add(TextContent.from(prompt + "\n"));
        for (int i = 1; i < end; i++) {
            Path image = Path.of(args[i]);
            content.add(TextContent.from("Image " + i + ":\n"));
            content.add(ImageContent.from(image.toUri()));
            System.err.printf("loaded image %d: %s%n", i, image);
        }

        try (var model = JinferChatModel.builder()
                .model(modelRef)
                .companion("media", mediaRef)
                .contextLength(8192)
                .maxOutputTokens(400)
                .thinking(false)
                .build()) {
            System.out.println("\n=== Gemma 4 says ===");
            System.out.println(model.chat(UserMessage.from(content)).aiMessage().text());
        }
    }
}
