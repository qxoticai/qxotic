package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.boundary.Arenas;
import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jinfer.boundary.media.ImageCodec;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.llm.Sampling;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Manual, real-model visual acceptance matrix; answers and generated fixtures are test artifacts.
 */
@Tag("driver")
final class Lfm2VisionAcceptanceDriver {
    private static final String TEXT = "hf.co/LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M";
    private static final String PROJECTOR =
            "hf.co/LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf";
    private static final Sampling GREEDY = new Sampling(0f, 1f, 0, 0f, 0L);
    private static final Pattern BBOX =
            Pattern.compile(
                    "\\\"bbox\\\"\\s*:\\s*\\[([0-9.]+),\\s*([0-9.]+),\\s*([0-9.]+),\\s*([0-9.]+)]");

    private record Answer(String text, Message reply) {}

    @Test
    void runOfficialCapabilityMatrix() throws Exception {
        Path candy = officialImage("candyImage", "/tmp/lfm2-vl-acceptance/candy.jpg");
        Path coco = officialImage("cocoImage", "/tmp/lfm2-vl-acceptance/coco_sample.png");
        Path audit = officialImage("auditImage", "/tmp/lfm2-vl-acceptance/audit-logs.png");

        Path text = com.qxotic.jinfer.testkit.TestModels.require(TEXT);
        Path projector = com.qxotic.jinfer.testkit.TestModels.require(PROJECTOR);
        try (Arena weights = Arenas.newCrossThread()) {
            var loaded = Models.load(text, weights, Map.of("media", projector));
            var cache =
                    PromptCache.Options.DEFAULTS
                            .withRetainedSessions(0)
                            .withBlockBudget(0)
                            .withContextCapacity(8192);
            ChatEngine engine = new ChatEngine(loaded, "lfm2-vl-official", cache);
            try {
                Media.Image candyImage = ImageCodec.load(candy);
                Media.Image cocoImage = ImageCodec.load(coco);
                Media.Image auditImage = ImageCodec.load(audit);

                String caption =
                        run(
                                        engine,
                                        "official-single-image",
                                        List.of(
                                                new Message(
                                                        Role.USER,
                                                        List.of(
                                                                new Content.Media(cocoImage),
                                                                new Content.Text(
                                                                        "Describe this image in two"
                                                                                + " concise"
                                                                                + " sentences.")))),
                                        List.of(),
                                        256)
                                .text()
                                .toLowerCase();
                assertTrue(caption.contains("cat"));
                assertTrue(caption.contains("couch") || caption.contains("sofa"));

                String multi =
                        run(
                                        engine,
                                        "official-multi-image",
                                        List.of(
                                                new Message(
                                                        Role.USER,
                                                        List.of(
                                                                new Content.Text("Media-1\n"),
                                                                new Content.Media(candyImage),
                                                                new Content.Text("\nMedia-2\n"),
                                                                new Content.Media(cocoImage),
                                                                new Content.Text(
                                                                        "\n"
                                                                            + "Caption Media-1 and"
                                                                            + " Media-2 separately."
                                                                            + " Keep each caption"
                                                                            + " to one"
                                                                            + " sentence.")))),
                                        List.of(),
                                        256)
                                .text()
                                .toLowerCase();
                assertTrue(multi.contains("media-1") && multi.contains("media-2"));
                assertTrue(multi.contains("candy") || multi.contains("m&m"));
                assertTrue(multi.contains("cat"));

                String ocr =
                        run(
                                        engine,
                                        "official-ocr",
                                        List.of(
                                                new Message(
                                                        Role.USER,
                                                        List.of(
                                                                new Content.Media(auditImage),
                                                                new Content.Text(
                                                                        "Read this audit log"
                                                                            + " screenshot."
                                                                            + " Identify the page"
                                                                            + " heading, table"
                                                                            + " columns, visible"
                                                                            + " rows, users,"
                                                                            + " actions,"
                                                                            + " timestamps, and"
                                                                            + " other structured"
                                                                            + " fields. Transcribe"
                                                                            + " the visible text in"
                                                                            + " reading order.")))),
                                        List.of(),
                                        512)
                                .text()
                                .toLowerCase();
                assertTrue(ocr.contains("audit log"));
                assertTrue(ocr.contains("clboetticher"));
                assertTrue(ocr.contains("repo.delete"));

                String layout =
                        run(
                                        engine,
                                        "official-document-layout",
                                        List.of(
                                                new Message(
                                                        Role.USER,
                                                        List.of(
                                                                new Content.Media(auditImage),
                                                                new Content.Text(layoutPrompt())))),
                                        List.of(),
                                        1024)
                                .text();
                assertTrue(layout.contains("image_index=0"));
                assertTrue(layout.contains("["));
                assertTrue(layout.toLowerCase().contains("audit log"));

                Answer grounding =
                        run(
                                engine,
                                "official-object-grounding",
                                List.of(
                                        Message.system(groundingSystemPrompt()),
                                        new Message(
                                                Role.USER,
                                                List.of(
                                                        new Content.Media(cocoImage),
                                                        new Content.Text(
                                                                "Provide bounding boxes for the two"
                                                                        + " cats and the two remote"
                                                                        + " controls.")))),
                                List.of(),
                                256);
                Json.parse(grounding.text());
                assertEquals(4, count(grounding.text(), "\"bbox_2d\""));
                assertTrue(grounding.text().toLowerCase().contains("cat"));
                assertTrue(grounding.text().toLowerCase().contains("remote"));

                Answer toolCall =
                        run(
                                engine,
                                "official-vision-tool-call",
                                List.of(
                                        new Message(
                                                Role.USER,
                                                List.of(
                                                        new Content.Media(cocoImage),
                                                        new Content.Text(
                                                                "Find a care guide for the animals"
                                                                    + " in this image. Choose the"
                                                                    + " best tool for this"
                                                                    + " request.")))),
                                tools(),
                                256);
                Content.ToolCall call =
                        assertInstanceOf(
                                Content.ToolCall.class, toolCall.reply().content().getFirst());
                assertEquals("search_pet_care", call.name());
                assertEquals("cat", String.valueOf(call.arguments().get("animal")).toLowerCase());
            } finally {
                engine.close();
            }
        }
    }

    @Test
    void runAcceptanceMatrix() throws Exception {
        Path output = Path.of("target", "lfm2-vl-acceptance");
        Files.createDirectories(output);
        Path ocr = write(output.resolve("ocr.png"), ocr());
        Path document = write(output.resolve("document.png"), document());
        Path scene = write(output.resolve("scene.png"), scene());
        Path grounding = write(output.resolve("grounding.png"), grounding());
        Path tiled = write(output.resolve("tiled-ocr.png"), tiledOcr());
        Path candy =
                Path.of(
                        System.getProperty(
                                "jinfer.lfm2.acceptanceImage",
                                "/tmp/lfm2-vl-acceptance/candy.jpg"));
        assertTrue(Files.isRegularFile(candy), () -> "Missing acceptance image: " + candy);

        Path text = com.qxotic.jinfer.testkit.TestModels.require(TEXT);
        Path projector = com.qxotic.jinfer.testkit.TestModels.require(PROJECTOR);
        try (Arena weights = Arenas.newCrossThread()) {
            var loaded = Models.load(text, weights, Map.of("media", projector));
            var cache =
                    PromptCache.Options.DEFAULTS
                            .withRetainedSessions(0)
                            .withBlockBudget(0)
                            .withContextCapacity(4096);
            ChatEngine engine = new ChatEngine(loaded, "lfm2-vl-acceptance", cache);
            try {
                if (Boolean.getBoolean("jinfer.lfm2.tiledOnly")) {
                    assertEquals(
                            "Z9Q4M2",
                            run(
                                    engine,
                                    "tiled-ocr",
                                    tiled,
                                    "Read the code printed near the far-right edge. Return only the"
                                            + " code.",
                                    32));
                    return;
                }
                assertEquals(
                        "A7K2Q9",
                        run(
                                engine,
                                "ocr",
                                ocr,
                                "Read the code in the image. Return only the code.",
                                32));
                assertEquals(
                        "42",
                        run(
                                engine,
                                "document-ocr",
                                document,
                                "Read the small table. Return only the value in the BETA row.",
                                32));
                assertEquals(
                        "red circles=3; blue squares=2; below=square",
                        run(
                                        engine,
                                        "recognition-count-spatial",
                                        scene,
                                        "Count the red circles and blue squares, then identify what"
                                                + " is directly below the middle red circle. Answer"
                                                + " exactly: red circles=N; blue squares=N;"
                                                + " below=SHAPE",
                                        64)
                                .toLowerCase());
                String grounded =
                        run(
                                engine,
                                "grounding",
                                grounding,
                                "Detect all instances of: red rectangle. Response must be a JSON"
                                    + " array: [{\"label\": ..., \"bbox\": [x1, y1, x2, y2]}, ...]."
                                    + " Coordinates are normalized to [0,1].",
                                96);
                double[] box = bbox(grounded);
                double iou = iou(box, new double[] {0.10, 0.20, 0.40, 0.60});
                assertTrue(grounded.contains("red rectangle"));
                assertTrue(iou >= 0.5, () -> "grounding IoU=" + iou + ": " + grounded);
                boolean strictContract =
                        box[4] == 1
                                && grounded.indexOf("\"label\"")
                                        == grounded.lastIndexOf("\"label\"");
                System.out.printf(
                        "ACCEPTANCE_CHECK\tgrounding-localization=PASS\tiou=%.3f"
                                + "\tgrounding-contract=%s%n",
                        iou, strictContract ? "PASS" : "FAIL");
                assertEquals(
                        "Z9Q4M2",
                        run(
                                engine,
                                "tiled-ocr",
                                tiled,
                                "Read the code printed near the far-right edge. Return only the"
                                        + " code.",
                                32));
                String natural =
                        run(
                                engine,
                                "natural-recognition",
                                candy,
                                "How many candy pieces are on the hand, and what are their colors?"
                                        + " Answer in one short sentence.",
                                64);
                assertTrue(natural.toLowerCase().contains("five"));
                assertTrue(natural.toLowerCase().contains("two teal"));
                assertTrue(natural.toLowerCase().contains("two orange"));
                assertTrue(natural.toLowerCase().contains("one green"));
            } finally {
                engine.close();
            }
        }
    }

    private static String run(
            ChatEngine engine, String name, Path imagePath, String prompt, int maxTokens)
            throws Exception {
        Media.Image image = ImageCodec.load(imagePath);
        Message message =
                new Message(Role.USER, List.of(new Content.Media(image), new Content.Text(prompt)));
        return run(engine, name, List.of(message), List.of(), maxTokens).text();
    }

    private static Answer run(
            ChatEngine engine,
            String name,
            List<Message> messages,
            List<Tool> tools,
            int maxTokens) {
        ChatEngine.Request request =
                new ChatEngine.Request(
                        messages,
                        tools,
                        false,
                        maxTokens,
                        null,
                        Duration.ZERO,
                        GREEDY,
                        null,
                        null,
                        List.of(),
                        null);
        long start = System.nanoTime();
        try (ChatEngine.Prepared prepared = engine.prepare(request)) {
            long preparedAt = System.nanoTime();
            ChatEngine.Completion completion = engine.complete(prepared, ChatEngine.ReplySink.NONE);
            long completedAt = System.nanoTime();
            double prepareMs = (preparedAt - start) / 1_000_000.0;
            double decodeMs = completion.result().decodeTime().toNanos() / 1_000_000.0;
            double promptIngestMs = (completedAt - preparedAt) / 1_000_000.0 - decodeMs;
            String answer = completion.reply().text().replace('\n', ' ').strip();
            System.out.printf(
                    "ACCEPTANCE\t%s\timage=%s\tpromptPositions=%d\tcompletionTokens=%d"
                            + "\tprepareMs=%.1f\tpromptIngestMs=%.1f\tdecodeMs=%.1f\ttotalMs=%.1f"
                            + "\t%s%n",
                    name,
                    media(messages),
                    prepared.promptTokens(),
                    completion.result().completionTokens(),
                    prepareMs,
                    promptIngestMs,
                    decodeMs,
                    (completedAt - start) / 1_000_000.0,
                    answer.isEmpty() ? completion.reply().content() : answer);
            return new Answer(answer, completion.reply());
        }
    }

    private static String media(List<Message> messages) {
        List<String> dimensions = new java.util.ArrayList<>();
        for (Message message : messages)
            for (Content content : message.content())
                if (content instanceof Content.Media value
                        && value.value() instanceof Media.Image image)
                    dimensions.add(image.width() + "x" + image.height());
        return String.join("+", dimensions);
    }

    private static int count(String text, String needle) {
        int count = 0;
        for (int at = 0; (at = text.indexOf(needle, at)) >= 0; at += needle.length()) count++;
        return count;
    }

    private static Path officialImage(String propertySuffix, String fallback) {
        Path path = Path.of(System.getProperty("jinfer.lfm2." + propertySuffix, fallback));
        assertTrue(Files.isRegularFile(path), () -> "Missing official test image: " + path);
        return path;
    }

    private static String groundingSystemPrompt() {
        return """
        When asked for bounding boxes for objects, return a valid JSON array.
        Each array item must be an object with:
        - image_id: the 0-based index of the image
        - bbox_2d: [xmin, ymin, xmax, ymax] normalized integer coordinates in [0, 1000]
        - label: a concise label you choose for the predicted object or region

        Return one item per visible matching object or region. Return [] if none are visible.\
        """;
    }

    private static String layoutPrompt() {
        return """
        Parse this document into its layout regions. The pages are provided as images in reading order. For every region, in reading order across all pages, output a header line immediately followed by the region's content:

        image_index=<n> <label> [xmin, ymin, xmax, ymax]
        <content>

        where:
        - image_index is the zero-based index of the page image the region appears on (0 for the first image, 1 for the second, and so on)
        - <label> is one of these layout labels: text, title, list, table, table_caption, table_footnote, image, image_block, image_caption, image_footnote, chart, equation, formula_number, code, code_caption, algorithm, aside_text, ref_text, phonetic, page_header, page_footer, page_number, page_footnote
        - [xmin, ymin, xmax, ymax] are normalized integer coordinates in [0, 1000]
        - <content> is the region's content: plain text for text regions, LaTeX for equations, OTSL for tables, and a short description for images and charts

        Separate each region block with one blank line. Return only the parsed regions.
        """;
    }

    private static List<Tool> tools() {
        return List.of(
                new Tool(
                        "search_pet_care",
                        Map.of(
                                "name",
                                "search_pet_care",
                                "description",
                                "Search for care guidance for an animal visible in an image.",
                                "parameters",
                                Map.of(
                                        "type",
                                        "object",
                                        "properties",
                                        Map.of(
                                                "animal",
                                                Map.of("type", "string"),
                                                "topic",
                                                Map.of("type", "string")),
                                        "required",
                                        List.of("animal", "topic")))),
                new Tool(
                        "search_replacement_remote",
                        Map.of(
                                "name",
                                "search_replacement_remote",
                                "description",
                                "Search for replacement remote controls visible in an image.",
                                "parameters",
                                Map.of(
                                        "type",
                                        "object",
                                        "properties",
                                        Map.of(
                                                "item",
                                                Map.of("type", "string"),
                                                "quantity",
                                                Map.of("type", "integer")),
                                        "required",
                                        List.of("item")))));
    }

    /** Returns normalized coordinates plus the source scale (1 or 1000). */
    private static double[] bbox(String answer) {
        Matcher matcher = BBOX.matcher(answer);
        assertTrue(matcher.find(), () -> "No bbox in: " + answer);
        double[] box = new double[5];
        for (int i = 0; i < 4; i++) box[i] = Double.parseDouble(matcher.group(i + 1));
        box[4] = box[2] > 1 || box[3] > 1 ? 1000 : 1;
        for (int i = 0; i < 4; i++) box[i] /= box[4];
        return box;
    }

    private static double iou(double[] a, double[] b) {
        double intersection =
                Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
                        * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        double areaA = (a[2] - a[0]) * (a[3] - a[1]);
        double areaB = (b[2] - b[0]) * (b[3] - b[1]);
        return intersection / (areaA + areaB - intersection);
    }

    private static BufferedImage canvas(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, width, height);
        graphics.dispose();
        return image;
    }

    private static Graphics2D graphics(BufferedImage image) {
        Graphics2D graphics = image.createGraphics();
        graphics.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setRenderingHint(
                RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        return graphics;
    }

    private static BufferedImage ocr() {
        BufferedImage image = canvas(512, 512);
        Graphics2D graphics = graphics(image);
        graphics.setColor(Color.BLACK);
        graphics.setFont(new Font(Font.MONOSPACED, Font.BOLD, 62));
        graphics.drawString("A7K2Q9", 125, 285);
        graphics.dispose();
        return image;
    }

    private static BufferedImage document() {
        BufferedImage image = canvas(512, 512);
        Graphics2D graphics = graphics(image);
        graphics.setColor(Color.BLACK);
        graphics.setStroke(new BasicStroke(3));
        graphics.drawRect(72, 90, 368, 300);
        graphics.drawLine(72, 190, 440, 190);
        graphics.drawLine(72, 290, 440, 290);
        graphics.drawLine(285, 90, 285, 390);
        graphics.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 34));
        graphics.drawString("ITEM", 115, 153);
        graphics.drawString("VALUE", 305, 153);
        graphics.drawString("ALPHA", 105, 253);
        graphics.drawString("17", 340, 253);
        graphics.drawString("BETA", 115, 353);
        graphics.drawString("42", 340, 353);
        graphics.dispose();
        return image;
    }

    private static BufferedImage scene() {
        BufferedImage image = canvas(512, 512);
        Graphics2D graphics = graphics(image);
        graphics.setColor(new Color(220, 35, 35));
        for (int x : new int[] {100, 256, 412}) graphics.fillOval(x - 45, 65, 90, 90);
        graphics.setColor(new Color(30, 80, 220));
        graphics.fillRect(206, 275, 100, 100);
        graphics.fillRect(362, 360, 100, 100);
        graphics.dispose();
        return image;
    }

    private static BufferedImage grounding() {
        BufferedImage image = canvas(512, 512);
        Graphics2D graphics = graphics(image);
        graphics.setColor(new Color(220, 35, 35));
        graphics.fillRect(51, 102, 154, 205); // approximately [0.10, 0.20, 0.40, 0.60]
        graphics.setColor(new Color(30, 80, 220));
        graphics.fillOval(330, 330, 100, 100);
        graphics.dispose();
        return image;
    }

    private static BufferedImage tiledOcr() {
        BufferedImage image = canvas(1800, 700);
        Graphics2D graphics = graphics(image);
        graphics.setColor(new Color(235, 240, 248));
        for (int x = 0; x < 1800; x += 100) graphics.drawLine(x, 0, x, 700);
        graphics.setColor(Color.BLACK);
        graphics.setFont(new Font(Font.MONOSPACED, Font.BOLD, 76));
        graphics.drawString("Z9Q4M2", 1460, 385);
        graphics.dispose();
        return image;
    }

    private static Path write(Path path, BufferedImage image) throws Exception {
        ImageIO.write(image, "png", path.toFile());
        return path;
    }
}
