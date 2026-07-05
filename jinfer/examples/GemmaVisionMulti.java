///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx24g
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-gemma4:0.1.0-SNAPSHOT

// Gemma 4 with MULTIPLE images in one prompt (per the docs: several image blocks per turn).
// Each image becomes its own <|image>...<image|> soft-token span; the model reasons across all of them.
//
//   Install once:  cd jinfer && ./mvnw -q -DskipTests install
//   Run (12B is worth it for cross-image reasoning):
//     jbang GemmaVisionMulti.java "Which image has more animals, and by how many?" a.jpg b.jpg \
//         ~/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf \
//         ~/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf
//   Defaults to E2B if no model paths are given.

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.llm.Gemma4;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class GemmaVisionMulti {

    static final String MODELS = System.getProperty("user.home") + "/Desktop/playground/models/unsloth/";

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: GemmaVisionMulti <prompt> <image1> <image2> [image3 ...] [textGguf mmprojGguf]");
            System.exit(2);
        }
        String prompt = args[0];

        // Trailing two args are the model paths if they end in .gguf; everything between is images.
        int end = args.length;
        Path textGguf = Path.of(MODELS + "gemma-4-E2B-it-Q8_0.gguf");
        Path mmproj   = Path.of(MODELS + "gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        if (end >= 3 && args[end - 1].endsWith(".gguf") && args[end - 2].endsWith(".gguf")) {
            textGguf = Path.of(args[end - 2]);
            mmproj   = Path.of(args[end - 1]);
            end -= 2;
        }

        Gemma4 model = Gemma4.loadModel(textGguf, mmproj, 8192);   // more context: many image tokens
        TurnTemplate template = model.turnTemplate().orElseThrow();
        Set<Integer> stops = model.stopTokens();

        // Build one user turn whose content interleaves the prompt text with N image parts.
        // Message content is an ordered List<Part>: Text then a Blob per image.
        List<Part> content = new ArrayList<>();
        content.add(new Part.Text(prompt + "\n"));
        for (int i = 1; i < end; i++) {
            Media.Image img = ImageCodec.load(Path.of(args[i]));
            content.add(new Part.Text("Image " + i + ":\n"));
            content.add(new Part.Blob(img));
            System.err.printf("loaded image %d: %s (%dx%d)%n", i, args[i], img.width(), img.height());
        }
        Message turn = new Message(Role.USER, content);

        List<Batch> batches = new ArrayList<>(template.conversationStart());
        batches.addAll(template.encodeTurn(turn));           // encodes every image inline, in order
        batches.addAll(template.generationPrompt(false));

        Gemma4.State state = model.newState(8192, 512);
        for (Batch b : Batch.prepare(batches, 512)) model.ingest(state, b);

        System.out.println("\n=== Gemma 4 says ===");
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
