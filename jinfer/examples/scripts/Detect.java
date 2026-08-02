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
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.media.ImageCodec;
import com.qxotic.jinfer.models.gemma4.Gemma4;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Detect {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) { System.err.println("usage: Detect <image> [labels] [gguf] [mmproj]"); System.exit(2); }
        Path image  = Path.of(args[0]);
        String what = args.length > 1 ? args[1] : "every prominent object";
        Path gguf   = Models.gemmaVision(args, 2);
        Path mmproj = Models.gemmaMmproj(args, 3);

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

        var reply = new StringBuilder();
        int tok = model.logits(state).argmax();
        for (int n = 0; n < 512 && !model.stopTokens().contains(tok); n++) {
            reply.append(model.tokenizer().decode(new int[] {tok}));
            model.ingest(state, Batch.step(tok));
            tok = model.logits(state).argmax();
        }
        System.out.println(reply);
        draw(image, img, reply.toString(), Path.of("detected.png"));
    }

    /** Rescale Gemma's 0-1024 boxes onto the real pixels and stroke them with their labels. */
    private static void draw(Path source, Media.Image img, String json, Path out) throws Exception {
        BufferedImage canvas = ImageIO.read(source.toFile());
        Graphics2D g = canvas.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(Math.max(2f, canvas.getWidth() / 400f)));
        g.setFont(g.getFont().deriveFont(Font.BOLD, Math.max(14f, canvas.getWidth() / 45f)));

        Color[] palette = {Color.RED, Color.CYAN, Color.YELLOW, Color.GREEN, Color.MAGENTA, Color.ORANGE};
        // Field ORDER is not guaranteed - the model emits box_2d first as often as label - so match
        // each object, then pull the two fields out of it independently.
        Matcher object = Pattern.compile("\\{[^{}]*\\}").matcher(json);
        Pattern label = Pattern.compile("\"label\"\\s*:\\s*\"([^\"]*)\"");
        Pattern box = Pattern.compile(
                "\"box_2d\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]");
        int found = 0;
        while (object.find()) {
            Matcher b = box.matcher(object.group());
            if (!b.find()) continue;
            Matcher l = label.matcher(object.group());
            String name = l.find() ? l.group(1) : "?";
            int ymin = scale(b.group(1), canvas.getHeight()), xmin = scale(b.group(2), canvas.getWidth());
            int ymax = scale(b.group(3), canvas.getHeight()), xmax = scale(b.group(4), canvas.getWidth());
            g.setColor(palette[found++ % palette.length]);
            g.drawRect(xmin, ymin, xmax - xmin, ymax - ymin);
            g.drawString(name, xmin + 4, Math.max(ymin + g.getFont().getSize(), 16));
        }
        g.dispose();
        ImageIO.write(canvas, "png", out.toFile());
        System.out.printf("%n%d box(es) drawn on %dx%d -> %s%n",
                found, canvas.getWidth(), canvas.getHeight(), out.toAbsolutePath());
    }

    private static int scale(String normalized, int pixels) {
        return Math.round(Integer.parseInt(normalized) / 1024f * pixels);
    }
}
