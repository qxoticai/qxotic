///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-gemma4:0.1.0-SNAPSHOT

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
// Tune sampling with -Djinfer.video.fps (default 1) and -Djinfer.video.maxFrames (default 16).

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.media.VideoCodec;
import com.qxotic.llm.Gemma4;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class GemmaVideo {

    static final String MODELS = System.getProperty("user.home") + "/Desktop/playground/models/unsloth/";

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: GemmaVideo <video> [prompt] [textGguf mmprojGguf]");
            System.exit(2);
        }
        Path video   = Path.of(args[0]);
        String prompt = args.length > 1 ? args[1] : "Describe this video.";
        Path textGguf = Path.of(args.length > 3 ? args[2] : MODELS + "gemma-4-E2B-it-Q8_0.gguf");
        Path mmproj   = Path.of(args.length > 3 ? args[3] : MODELS + "gemma-4-E2B-it-GGUF/mmproj-F32.gguf");

        int fps       = Integer.getInteger("jinfer.video.fps", VideoCodec.DEFAULT_FPS);
        int maxFrames = Integer.getInteger("jinfer.video.maxFrames", VideoCodec.DEFAULT_MAX_FRAMES);

        // 1. Sample the video -> Media.Video (frames + fps). ffmpeg does the demux/decode.
        Media.Video vid = VideoCodec.load(video, fps, maxFrames);
        System.err.printf("sampled %d frames @ %d fps (%dx%d)%n", vid.frames().length, fps,
                vid.frames()[0].width(), vid.frames()[0].height());

        Gemma4 model = Gemma4.loadModel(textGguf, mmproj, 8192);   // frame tokens need headroom
        TurnTemplate template = model.turnTemplate().orElseThrow();
        Set<Integer> stops = model.stopTokens();

        // 2. One user turn carrying the prompt + the whole Media.Video. encodeTurn expands it into
        //    timestamped frame image-blocks and runs each frame through the vision encoder.
        List<Batch> batches = new ArrayList<>(template.conversationStart());
        batches.addAll(template.encodeTurn(Message.user(prompt, vid)));
        batches.addAll(template.generationPrompt(false));

        Gemma4.State state = model.newState(8192, 512);
        int pos = 0;
        for (Batch b : Batch.prepare(batches, 512)) { model.ingest(state, b); pos += b.count(); }
        System.err.printf("prompt: %d positions%n", pos);

        System.out.println("\n=== Gemma 4 describes the video ===");
        int tok = model.logits(state).argmax();
        for (int n = 0; n < 400 && !stops.contains(tok); n++) {
            System.out.print(model.tokenizer().decode(tok));
            System.out.flush();
            model.ingest(state, Batch.step(tok));
            tok = model.logits(state).argmax();
        }
        System.out.println();
    }
}
