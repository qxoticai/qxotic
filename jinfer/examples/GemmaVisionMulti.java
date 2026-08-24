///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-gemma4
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Ask Gemma 4 to compare multiple images in one prompt.
//   jbang GemmaVisionMulti.java "Which image has more animals?" a.jpg b.jpg

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class GemmaVisionMulti {

    private static final String DEFAULT_MODEL =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";
    private static final String DEFAULT_MEDIA =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf";

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("usage: GemmaVisionMulti <prompt> <image1> <image2> [image3 ...]");
            System.exit(2);
        }
        String prompt = args[0];

        List<Content> content = new ArrayList<>();
        content.add(TextContent.from(prompt + "\n"));
        for (int i = 1; i < args.length; i++) {
            Path image = Path.of(args[i]);
            content.add(TextContent.from("Image " + i + ":\n"));
            content.add(ImageContent.from(image.toUri()));
        }

        try (var model = JinferChatModel.builder()
                .model(DEFAULT_MODEL)
                .companion("media", DEFAULT_MEDIA)
                .contextLength(8192)
                .maxOutputTokens(400)
                .thinking(false)
                .build()) {
            System.out.println(model.chat(UserMessage.from(content)).aiMessage().text());
        }
    }
}
