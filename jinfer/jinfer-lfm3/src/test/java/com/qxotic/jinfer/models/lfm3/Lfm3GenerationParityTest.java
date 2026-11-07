package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * End to end against llama-server on the same Q8_0 checkpoints
 * (jinfer/scripts/lfm3-chat-reference.py): the prompt our template frames for each conversation
 * must be the server's token for token, with media rows exactly where mtmd puts them, and our
 * greedy continuation must be the server's - until a step where the server itself saw a near-tie,
 * past which the two quantized stacks may part.
 */
@Tag("integration")
final class Lfm3GenerationParityTest {

    private static final String MODELS = "hf.co/LiquidAI/";

    /** The server's request order: tojson keeps a map's insertion order. */
    private static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    ordered(
                            "type",
                            "function",
                            "function",
                            ordered(
                                    "name",
                                    "get_weather",
                                    "description",
                                    "Current weather for a city",
                                    "parameters",
                                    ordered(
                                            "type",
                                            "object",
                                            "properties",
                                            Map.of("city", Map.of("type", "string")),
                                            "required",
                                            List.of("city")))));

    private static Map<String, Object> ordered(Object... pairs) {
        Map<String, Object> map = new java.util.LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) map.put((String) pairs[i], pairs[i + 1]);
        return map;
    }

    @ParameterizedTest
    @CsvSource({
        "text, LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf, ",
        "vl, LFM3-VL-2B-GGUF/LFM3-VL-2B-Q8_0.gguf, LFM3-VL-2B-GGUF/mmproj-LFM3-VL-2B-F16.gguf",
        "omni, LFM3-Omni-2B-GGUF/LFM3-Omni-2B-Q8_0.gguf,"
                + " LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf"
    })
    @SuppressWarnings("unchecked")
    void promptsAndGreedyRepliesMatchLlamaServer(String kind, String text, String mmproj)
            throws Exception {
        Path file = Oracle.dir("chat").resolve(kind + ".json");
        Map<String, Object> cases = (Map<String, Object>) Json.parse(Files.readString(file));
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model =
                    mmproj == null
                            ? Lfm3.loadModel(TestModels.require(MODELS + text), arena)
                            : Lfm3.loadModel(
                                    TestModels.require(MODELS + text),
                                    TestModels.require(MODELS + mmproj),
                                    arena);
            Lfm3ChatTemplate template = new Lfm3ChatTemplate(model);
            int vocab = model.configuration().vocabularySize();
            int visionOpen = model.tokenizer().vocabulary().id("<|vision_start|>");
            int visionClose = model.tokenizer().vocabulary().id("<|vision_end|>");
            for (Map.Entry<String, Object> entry : cases.entrySet()) {
                Map<String, Object> c = (Map<String, Object>) entry.getValue();
                Conversation conversation = conversation(c);
                int positions = ((Number) c.get("prompt_positions")).intValue();

                try (Lfm3.State state = model.newState(positions + 256, 512)) {
                    List<Integer> ours = new ArrayList<>();
                    template.encode(
                            conversation,
                            512,
                            batch -> {
                                if (batch.input() instanceof Batch.Input.Tokens t)
                                    for (int id : t.ids()) ours.add(id);
                                else for (int r = 0; r < batch.count(); r++) ours.add(-1);
                                model.ingest(state, batch);
                            });
                    // the server's pieces, mtmd framing each marker as <|vision_start|> rows
                    // <|vision_end|>
                    List<List<Number>> pieces = (List<List<Number>>) c.get("pieces");
                    int markers = pieces.size() - 1;
                    int rows =
                            markers == 0
                                    ? 0
                                    : (positions - textTokens(pieces) - 2 * markers) / markers;
                    List<Integer> expected = new ArrayList<>();
                    for (int p = 0; p < pieces.size(); p++) {
                        for (Number id : pieces.get(p)) expected.add(id.intValue());
                        if (p == markers) break;
                        expected.add(visionOpen);
                        for (int r = 0; r < rows; r++) expected.add(-1);
                        expected.add(visionClose);
                    }
                    assertArrayEquals(
                            expected.stream().mapToInt(Integer::intValue).toArray(),
                            ours.stream().mapToInt(Integer::intValue).toArray(),
                            entry.getKey() + ": prompt");
                    assertEquals(positions, ours.size(), entry.getKey() + ": positions");

                    List<Number> reference = (List<Number>) c.get("tokens");
                    List<List<Number>> top2 = (List<List<Number>>) c.get("top2");
                    int agreed = 0;
                    for (; agreed < reference.size(); agreed++) {
                        float[] logits =
                                Oracle.rows(
                                        Views.castToSegmentBacked(model.logits(state, 0), "logits"),
                                        1,
                                        vocab);
                        int want = reference.get(agreed).intValue(),
                                got = Oracle.argmax(logits, 0, vocab);
                        if (got != want) {
                            double gap =
                                    top2.get(agreed).get(0).doubleValue()
                                            - top2.get(agreed).get(1).doubleValue();
                            System.out.printf(
                                    "%s/%s: parts at step %d (server's top-2 gap %.4f)%n",
                                    kind, entry.getKey(), agreed, gap);
                            assertTrue(
                                    gap < 0.1,
                                    entry.getKey()
                                            + ": step "
                                            + agreed
                                            + " is no near-tie, gap "
                                            + gap);
                            break;
                        }
                        model.ingest(state, Batch.step(got));
                    }
                    System.out.printf(
                            "%s/%s: %d prompt positions, %d/%d generated tokens identical%n",
                            kind, entry.getKey(), positions, agreed, reference.size());
                    assertTrue(
                            agreed >= Math.min(8, reference.size()),
                            entry.getKey() + ": only " + agreed + " tokens agree");
                }
            }
        }
    }

    private static int textTokens(List<List<Number>> pieces) {
        int n = 0;
        for (List<Number> piece : pieces) n += piece.size();
        return n;
    }

    @SuppressWarnings("unchecked")
    private static Conversation conversation(Map<String, Object> c) throws Exception {
        List<Message> messages = new ArrayList<>();
        if (c.get("system") instanceof String system) messages.add(Message.system(system));
        List<Object> user = (List<Object>) c.get("user");
        List<List<String>> media = (List<List<String>>) c.get("media"); // [kind, path]
        List<Content> parts = new ArrayList<>();
        int m = 0;
        for (Object part : user) {
            if (part instanceof String s) {
                parts.add(new Content.Text(s));
                continue;
            }
            Path path = Oracle.ROOT.resolve("../..").resolve(media.get(m).get(1));
            parts.add(
                    new Content.Media(
                            media.get(m).get(0).equals("image")
                                    ? ImageCodec.load(path)
                                    : Lfm3AudioParityTest.wav(path),
                            null));
            m++;
        }
        messages.add(new Message(Role.USER, parts));
        return new Conversation(
                messages,
                Boolean.TRUE.equals(c.get("tools")) ? List.of(WEATHER) : List.of(),
                Boolean.TRUE.equals(c.get("thinking")));
    }
}
