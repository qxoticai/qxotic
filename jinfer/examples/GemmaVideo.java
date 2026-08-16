///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-xlangchain4j:0.1.0

// Gemma 4 video understanding (equivalent of the docs' "Describe this video."):
//   https://ai.google.dev/gemma/docs/capabilities/vision/video
// jinfer decodes the video to sampled frames (ffmpeg) and feeds them as timestamped image blocks
// ("00:00 <|image>…", "00:01 …") - Gemma's video-as-frames approach.
//
//   Install once:  cd jinfer && ./mvnw -q -DskipTests install
//   E2B:  jbang GemmaVideo.java clip.mp4
//   12B:  jbang GemmaVideo.java clip.mp4 "Describe this video." \
//             ~/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf \
//             ~/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf
//
// IMPORTANT: each frame is ~256 image tokens at the default budget, so many frames blow the context
// fast. Use a LOW per-frame budget for video:
//     jbang -Djinfer.gemma4.imageTokenBudget=140 GemmaVideo.java clip.mp4
// Tune sampling with -Djinfer.video.frames (default 16 here; frames are spread uniformly across
// the WHOLE duration, each stamped with its true timestamp).

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.x.boundary.media.VideoCodec;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.VideoContent;

import java.nio.file.Path;

public class GemmaVideo {

    static final String MODELS = System.getenv().getOrDefault("JINFER_MODELS_UNSLOTH", System.getProperty("user.home") + "/models/unsloth/");

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: GemmaVideo <video> [prompt] [textGguf mmprojGguf]");
            System.exit(2);
        }
        Path video   = Path.of(args[0]);
        String prompt = args.length > 1 ? args[1] : "Describe this video.";
        Path textGguf = Path.of(args.length > 3 ? args[2] : MODELS + "gemma-4-E2B-it-Q8_0.gguf");
        Path mmproj   = Path.of(args.length > 3 ? args[3] : MODELS + "gemma-4-E2B-it-GGUF/mmproj-F32.gguf");

        int numFrames = Integer.getInteger("jinfer.video.frames", 16);

        try (var model = JinferChatModel.builder()
                .modelPath(textGguf)
                .companion("media", mmproj)
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
