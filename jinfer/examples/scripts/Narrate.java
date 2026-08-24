///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-gemma4 com.qxotic:jinfer-inflect2
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Describe an image, then synthesize the description into narration.wav.
//   jbang Narrate.java photo.jpg
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferSpeechModel;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class Narrate {

    private static final String DEFAULT_VISION_MODEL =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";
    private static final String DEFAULT_MEDIA =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf";
    private static final String DEFAULT_SPEECH_MODEL =
            "hf.co/remixerdec/Inflect-Nano-v2-GGUF:Q8_0";

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.err.println(
                    "usage: Narrate <image> [vision-model-ref] [media-ref] [speech-model-ref]");
            System.exit(2);
        }

        Path image = Path.of(args[0]);
        String visionModelRef = args.length > 1 ? args[1] : DEFAULT_VISION_MODEL;
        String mediaRef = args.length > 2 ? args[2] : DEFAULT_MEDIA;
        String speechModelRef = args.length > 3 ? args[3] : DEFAULT_SPEECH_MODEL;

        String description;
        try (var eyes = JinferChatModel.builder()
                .model(visionModelRef)
                .companion("media", mediaRef)
                .maxOutputTokens(96)
                .thinking(false)
                .build()) {
            description = eyes.chat(UserMessage.from(
                            TextContent.from("Describe this image vividly in two sentences."),
                            ImageContent.from(image.toUri())))
                    .aiMessage()
                    .text();
        }

        System.out.println(description);
        try (var voice = JinferSpeechModel.builder().model(speechModelRef).build()) {
            byte[] wav = voice.synthesize(description).audio().binaryData();
            Files.write(Path.of("narration.wav"), wav);
            System.out.printf("%nWrote narration.wav (%.1f KB).%n", wav.length / 1024.0);
        }
    }
}
