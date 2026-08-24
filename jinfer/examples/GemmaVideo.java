///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-gemma4 com.qxotic:jinfer-codecs
//DEPS com.qxotic:jam-native com.qxotic:jam-vector

// Sample a video with ffmpeg and ask a local multimodal model to describe it.
//   jbang GemmaVideo.java clip.mp4 "Summarize the main events."

import com.qxotic.jinfer.codecs.VideoCodec;
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.VideoContent;

import java.nio.file.Path;

public class GemmaVideo {

    private static final String DEFAULT_MODEL =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";
    private static final String DEFAULT_MEDIA =
            "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf";

    public static void main(String[] args) {
        if (args.length == 0) {
            System.err.println("usage: GemmaVideo <video> [prompt] [model-ref] [media-ref]");
            System.exit(2);
        }
        Path video = Path.of(args[0]);
        String prompt = args.length > 1 ? args[1] : "Describe this video.";
        String modelRef = args.length > 2 ? args[2] : DEFAULT_MODEL;
        String mediaRef = args.length > 3 ? args[3] : DEFAULT_MEDIA;
        int frames = Integer.getInteger("jinfer.video.frames", 16);

        try (var model = JinferChatModel.builder()
                .model(modelRef)
                .companion("media", mediaRef)
                .videoSampler(path -> VideoCodec.ffmpeg().uniform(path, frames))
                .contextLength(8192)
                .maxOutputTokens(400)
                .thinking(false)
                .build()) {
            var message = UserMessage.from(
                    TextContent.from(prompt), VideoContent.from(video.toUri()));
            System.out.println(model.chat(message).aiMessage().text());
        }
    }
}
