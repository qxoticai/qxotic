///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//COMPILE_OPTIONS --enable-preview --release 25
//RUNTIME_OPTIONS --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx16g
//REPOS mavenLocal,central
//DEPS com.qxotic:jinfer-gemma4:0.1.0
//SOURCES Models.java

// Object detection with boxes DRAWN, not printed. Gemma 4 returns normalized 0-1024 coordinates as
// JSON; this rescales them to the image and writes an annotated PNG you can actually look at.
//
//   jbang Detect.java photo.jpg "person, dog, bicycle"
//   -> detected.png
//
// Detection is prompt-driven - the same model that describes an image also localizes in it, so
// there is no detector to train, load or wire up.
//
// USE A BIG MODEL FOR THIS ONE. Describing an image and LOCALIZING in it are very different asks:
// E2B labels correctly and places badly (asked for the llama and the mug in a test photo, it
// labelled both right and put the llama's box inside the mug). 12B places both correctly from the
// same prompt and the same code. Pass a smaller GGUF explicitly if you want to see it fail:
//     jbang Detect.java photo.jpg "a llama" ~/models/.../gemma-4-E2B-it-Q8_0.gguf ~/models/.../mmproj-F32.gguf
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.jinfer.models.gemma4.Gemma4;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Detect {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) { System.err.println("usage: Detect <image> [labels] [gguf] [mmproj]"); System.exit(2); }
        Path image  = Path.of(args[0]);
        String what = args.length > 1 ? args[1] : "every prominent object";
        Path gguf   = Models.gemmaDetect(args, 2);
        Path mmproj = Models.gemmaDetectMmproj(args, 3);

        var model = Gemma4.loadModel(gguf, mmproj, 4096, Arena.ofAuto());
        var template = model.turnTemplate().orElseThrow();
        Media.Image img = ImageCodec.load(image);

        String prompt = "Detect " + what + ". Output ONLY a JSON array, each element "
                + "{\"label\": string, \"box_2d\": [ymin, xmin, ymax, xmax]} with coordinates "
                + "normalized to 0-1024.";

        List<Batch> batches = new ArrayList<>(template.conversationStart());
        batches.addAll(template.encodeTurn(Message.user(prompt, img)));
        batches.addAll(template.generationPrompt(false));

        var state = model.newState(4096, 512);
        for (Batch b : Batch.prepare(batches, 512)) model.ingest(state, b);

        var stops = model.stopTokens();
        var ids = new ArrayList<Integer>();
        int tok = model.logits(state).argmax();
        for (int n = 0; n < 512 && !stops.contains(tok); n++) {
            ids.add(tok);
            model.ingest(state, Batch.step(tok));
            tok = model.logits(state).argmax();
        }
        String reply = model.tokenizer().decode(ids.stream().mapToInt(Integer::intValue).toArray());
        System.out.println(reply);
        draw(image, reply, Path.of("detected.png"));
    }

    /** Rescale Gemma's 0-1024 boxes onto the real pixels and stroke them with their labels. */
    private static void draw(Path source, String json, Path out) throws Exception {
        BufferedImage canvas = ImageIO.read(source.toFile());
        Graphics2D g = canvas.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(Math.max(2f, canvas.getWidth() / 400f)));
        g.setFont(g.getFont().deriveFont(Font.BOLD, Math.max(14f, canvas.getWidth() / 45f)));

        Color[] palette = {Color.RED, Color.CYAN, Color.YELLOW, Color.GREEN, Color.MAGENTA, Color.ORANGE};
        // A real parser, so field order does not matter - the model emits box_2d before label as
        // often as after. Models like to wrap the array in prose or a ```json fence; take the
        // outermost [ ... ] and let the parser do the rest.
        int from = json.indexOf('['), to = json.lastIndexOf(']');
        if (from < 0 || to < from) throw new IllegalStateException("no JSON array in reply:\n" + json);
        int found = 0;
        for (Object element : (List<?>) JsonCodec.parse(json.substring(from, to + 1))) {
            if (!(element instanceof Map<?, ?> object)) continue;
            if (!(object.get("box_2d") instanceof List<?> box) || box.size() != 4) continue;
            String name = object.get("label") instanceof String s ? s : "?";
            int ymin = scale(box.get(0), canvas.getHeight()), xmin = scale(box.get(1), canvas.getWidth());
            int ymax = scale(box.get(2), canvas.getHeight()), xmax = scale(box.get(3), canvas.getWidth());
            g.setColor(palette[found++ % palette.length]);
            g.drawRect(xmin, ymin, xmax - xmin, ymax - ymin);
            g.drawString(name, xmin + 4, Math.max(ymin + g.getFont().getSize(), 16));
        }
        g.dispose();
        ImageIO.write(canvas, "png", out.toFile());
        System.out.printf("%n%d box(es) drawn on %dx%d -> %s%n",
                found, canvas.getWidth(), canvas.getHeight(), out.toAbsolutePath());
    }

    private static int scale(Object normalized, int pixels) {
        return Math.round(((Number) normalized).floatValue() / 1024f * pixels);
    }
}
