///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-gemma4
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Ask a local multimodal model about an image.
//   jbang GemmaVision.java cat.jpg "What is in this image?"

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;

import java.nio.file.Path;

public class GemmaVision {

    private static final String DEFAULT_MODEL =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";
    private static final String DEFAULT_MEDIA =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf";

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("usage: GemmaVision <image> <prompt> [model-ref] [media-ref]");
            System.exit(2);
        }
        Path image = Path.of(args[0]);
        String prompt = args[1];
        String modelRef = args.length > 2 ? args[2] : DEFAULT_MODEL;
        String mediaRef = args.length > 3 ? args[3] : DEFAULT_MEDIA;

        try (var model = JinferChatModel.builder()
                .model(modelRef)
                .companion("media", mediaRef)
                .contextLength(4096)
                .maxOutputTokens(300)
                .thinking(false)
                .build()) {
            var message = UserMessage.from(
                    TextContent.from(prompt), ImageContent.from(image.toUri()));
            System.out.println(model.chat(message).aiMessage().text());
        }
    }
}
