///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -Xmx16g
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-gemma4
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
//DEPS com.qxotic:json

// Detect objects and draw their bounding boxes into detected.png.
//   jbang Detect.java photo.jpg "person, dog, bicycle"
import com.qxotic.format.json.Json;
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class Detect {

    private static final String DEFAULT_MODEL =
            "hf.co/unsloth/gemma-4-12b-it-GGUF:Q8_0";
    private static final String DEFAULT_MEDIA =
            "hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf";

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.err.println("usage: Detect <image> [labels] [model-ref] [media-ref]");
            System.exit(2);
        }
        Path image = Path.of(args[0]);
        String what = args.length > 1 ? args[1] : "every prominent object";
        String modelRef = args.length > 2 ? args[2] : DEFAULT_MODEL;
        String mediaRef = args.length > 3 ? args[3] : DEFAULT_MEDIA;

        String prompt = "Detect " + what + ". Return only a JSON array. Each element is "
                + "{\"label\": \"name\", \"box_2d\": [ymin, xmin, ymax, xmax]} with coordinates "
                + "normalized to 0-1000.";

        String reply;
        try (var model = JinferChatModel.builder()
                .model(modelRef)
                .companion("media", mediaRef)
                .contextLength(4096)
                .maxOutputTokens(512)
                .thinking(false)
                .build()) {
            var message = UserMessage.from(
                    TextContent.from(prompt), ImageContent.from(image.toUri()));
            reply = model.chat(message).aiMessage().text();
        }
        System.out.println(reply);
        draw(image, reply, Path.of("detected.png"));
    }

    private static void draw(Path source, String json, Path out) throws IOException {
        int from = json.indexOf('[');
        int to = json.lastIndexOf(']');
        if (from < 0 || to < from) throw new IllegalStateException("no JSON array in reply:\n" + json);
        List<?> detections = Json.parseList(json.substring(from, to + 1));

        BufferedImage canvas = ImageIO.read(source.toFile());
        if (canvas == null) throw new IOException("unsupported image: " + source);

        Color[] palette = {
            Color.RED, Color.CYAN, Color.YELLOW, Color.GREEN, Color.MAGENTA, Color.ORANGE
        };
        int found = 0;
        Graphics2D g = canvas.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setStroke(new BasicStroke(Math.max(2f, canvas.getWidth() / 400f)));
            g.setFont(g.getFont().deriveFont(Font.BOLD, Math.max(14f, canvas.getWidth() / 45f)));

            for (Object element : detections) {
                if (!(element instanceof Map<?, ?> object)) {
                    continue;
                }
                if (!(object.get("box_2d") instanceof List<?> box) || box.size() != 4) {
                    continue;
                }
                if (!(box.get(0) instanceof Number y0)
                        || !(box.get(1) instanceof Number x0)
                        || !(box.get(2) instanceof Number y1)
                        || !(box.get(3) instanceof Number x1)) {
                    continue;
                }

                int py0 = scale(y0, canvas.getHeight());
                int px0 = scale(x0, canvas.getWidth());
                int py1 = scale(y1, canvas.getHeight());
                int px1 = scale(x1, canvas.getWidth());
                int top = Math.min(py0, py1);
                int left = Math.min(px0, px1);
                int bottom = Math.max(py0, py1);
                int right = Math.max(px0, px1);
                String name = object.get("label") instanceof String s ? s : "?";
                g.setColor(palette[found++ % palette.length]);
                g.drawRect(left, top, right - left, bottom - top);
                g.drawString(name, left + 4, Math.max(top + g.getFont().getSize(), 16));
            }
        } finally {
            g.dispose();
        }
        ImageIO.write(canvas, "png", out.toFile());
        System.out.printf("%nDrew %d box(es) on %dx%d. Wrote %s%n",
                found, canvas.getWidth(), canvas.getHeight(), out.toAbsolutePath());
    }

    private static int scale(Number normalized, int pixels) {
        return Math.clamp(Math.round(normalized.floatValue() / 1000f * pixels), 0, pixels);
    }
}
