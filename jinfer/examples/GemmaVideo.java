///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --release 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-gemma4 com.qxotic:jinfer-codecs
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
//SOURCES scripts/Models.java

// Gemma 4 video understanding:
//   https://ai.google.dev/gemma/docs/capabilities/vision/video
// Jinfer samples timestamped frames with ffmpeg and sends them to the model as image content.
//
//   E2B:  jbang GemmaVideo.java clip.mp4
//   12B:  jbang GemmaVideo.java clip.mp4 "Describe this video." \
//             hf.co/unsloth/gemma-4-12b-it-GGUF:Q8_0 \
//             hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf
//
// Each frame consumes part of the context window. Use a lower image token budget for video:
//     jbang -Djinfer.gemma4.imageTokenBudget=140 GemmaVideo.java clip.mp4
// Set -Djinfer.video.frames to control sampling. The default is 16 frames spread uniformly across
// the video.

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.codecs.VideoCodec;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.VideoContent;

import java.nio.file.Path;

public class GemmaVideo {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println(
                    "usage: GemmaVideo <video> [prompt] [model-ref] [mmproj-ref]");
            System.exit(2);
        }
        Path video = Path.of(args[0]);
        String prompt = args.length > 1 ? args[1] : "Describe this video.";
        String modelRef = Models.gemma(args, 2);
        String mediaRef = Models.gemmaMmproj(args, 3);

        int numFrames = Integer.getInteger("jinfer.video.frames", 16);

        try (var model = JinferChatModel.builder()
                .model(modelRef)
                .companion("media", mediaRef)
                .videoSampler(path -> VideoCodec.ffmpeg().uniform(path, numFrames))
                .contextLength(8192)
                .maxOutputTokens(400)
                .thinking(false)
                .build()) {
            var message = UserMessage.from(
                    TextContent.from(prompt), VideoContent.from(video.toUri()));
            System.err.printf("sampling %d frames uniformly from %s%n", numFrames, video);
            System.out.println("\n=== Gemma 4 describes the video ===");
            System.out.println(model.chat(message).aiMessage().text());
        }
    }
}
