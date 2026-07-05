///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
// jinfer is a local (unpublished) build - install it to your ~/.m2 once, then jbang resolves it:
//     cd jinfer && ./mvnw -q -DskipTests install
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-gemma4:0.1.0-SNAPSHOT

// Gemma 4 vision (image -> text) through jinfer's multimodal chat API.
// Same code runs on every Gemma 4 size - only the GGUF + mmproj paths change:
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

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.llm.Gemma4;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class GemmaVision {

    static final String MODELS = System.getProperty("user.home") + "/Desktop/playground/models/unsloth/";

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: GemmaVision <image> <prompt> [textGguf] [mmprojGguf]");
            System.exit(2);
        }
        Path image  = Path.of(args[0]);
        String prompt = args[1];
        Path textGguf = Path.of(args.length > 2 ? args[2] : MODELS + "gemma-4-E2B-it-Q8_0.gguf");
        Path mmproj   = Path.of(args.length > 3 ? args[3] : MODELS + "gemma-4-E2B-it-GGUF/mmproj-F32.gguf");

        // 1. Load the text model WITH its vision projector. This is the whole multimodal setup.
        Gemma4 model = Gemma4.loadModel(textGguf, mmproj, 4096);
        TurnTemplate template = model.turnTemplate().orElseThrow();
        Set<Integer> stops = model.stopTokens();

        // 2. Decode the image to jinfer's universal RGB Media.Image (any format via ffmpeg/ImageIO).
        Media.Image img = ImageCodec.load(image);
        System.err.printf("image %dx%d, model %s%n", img.width(), img.height(), textGguf.getFileName());

        // 3. Build the prompt: one user turn carrying text + the image. encodeTurn runs the vision
        //    encoder and emits the image's soft-token embeddings inline - nothing else to wire.
        List<Batch> batches = new ArrayList<>(template.conversationStart());
        batches.addAll(template.encodeTurn(Message.user(prompt, img)));
        batches.addAll(template.generationPrompt(false));   // false = answer directly (no <think>)

        // 4. Ingest (prompt + image embeddings) and greedy-decode the reply.
        Gemma4.State state = model.newState(4096, 512);
        for (Batch b : Batch.prepare(batches, 512)) model.ingest(state, b);

        System.out.println("\n=== Gemma 4 says ===");
        int tok = model.logits(state).argmax();
        for (int n = 0; n < 300 && !stops.contains(tok); n++) {
            System.out.print(model.tokenizer().decode(tok));
            System.out.flush();
            model.ingest(state, Batch.step(tok));
            tok = model.logits(state).argmax();
        }
        System.out.println();
    }
}
