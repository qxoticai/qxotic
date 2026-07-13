// The codec round-trip law + decode parity, on the real LFM2 tokenizer (no weights): a reply
// token stream fed through the codec's ReplyDecoder and appended to the conversation must
// re-encode to EXACTLY encode(conversation) ++ reply ++ closeTurn ++ generationPrompt (verbatim
// splice = KV continuity), and the decoder must structure the reply identically to the retired
// Lfm2ToolCallDetector + think-demux path.
//   java ... com.qxotic.jinfer.models.lfm2.Lfm2CodecLawOracle [model.gguf]
package com.qxotic.jinfer.models.lfm2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyDecoder;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.GgufTokenizer;
import com.qxotic.jinfer.testkit.Checks;
import com.qxotic.toknroll.IntSequence;
import java.io.ByteArrayOutputStream;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

public final class Lfm2CodecLawOracle {

    private static final Checks checks = new Checks();

    public static void main(String[] args) throws Exception {
        Path model =
                Path.of(
                        args.length > 0
                                ? args[0]
                                : "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        if (!Files.exists(model)) {
            System.out.println("Lfm2CodecLawOracle: model not found (" + model + "), skipping");
            return;
        }
        GGUF g;
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            g = ModelLoader.readGguf(channel, model.toString());
        }
        GgufTokenizer tok = new GgufTokenizer(g);
        Lfm2ChatTemplate template = new Lfm2ChatTemplate(tok);
        int thinkOpen = tok.requiredSpecial("<think>");
        int thinkClose = tok.requiredSpecial("</think>");
        int tcStart = tok.requiredSpecial("<|tool_call_start|>");
        int tcEnd = tok.requiredSpecial("<|tool_call_end|>");
        IntSequence scaffold =
                IntSequence.of(tok.requiredSpecial("<|im_start|>"))
                        .concat(tok.encode("assistant\n"));
        IntSequence close =
                IntSequence.of(tok.requiredSpecial("<|im_end|>")).concat(tok.encode("\n"));

        // reply A: a thinking reply
        IntSequence replyA =
                IntSequence.of(thinkOpen)
                        .concat(tok.encode("easy arithmetic\n2+2=4"))
                        .concat(IntSequence.of(thinkClose))
                        .concat(tok.encode("\n\nThe answer is 4."));
        Conversation convA = new Conversation(List.of(Message.user("2+2?")));
        Message msgA = decode(template, replyA);
        checks.check(
                law(template, convA, msgA, replyA, close, scaffold),
                "round-trip law: thinking reply splices verbatim");
        checks.check(
                "easy arithmetic\n2+2=4".equals(reasoningText(msgA)),
                "decode parity: reasoning text");
        checks.check("\n\nThe answer is 4.".equals(msgA.text()), "decode parity: visible text");

        // reply B: text + a tool call
        Tool weather = Lfm2ToolOracle.WEATHER;
        IntSequence callPayload = tok.encode("[get_weather(city='Paris')]");
        IntSequence replyB =
                tok.encode("I will check.\n")
                        .concat(IntSequence.of(tcStart))
                        .concat(callPayload)
                        .concat(IntSequence.of(tcEnd));
        Conversation convB =
                new Conversation(
                        List.of(Message.user("Weather in Paris?")), List.of(weather), true, "");
        Message msgB = decode(template, replyB);
        checks.check(
                law(template, convB, msgB, replyB, close, scaffold),
                "round-trip law: tool-call reply splices verbatim");
        List<Part.ToolCall> callsB = toolCalls(msgB);
        checks.check(
                callsB.size() == 1
                        && "get_weather".equals(callsB.get(0).name())
                        && "Paris".equals(callsB.get(0).arguments().get("city")),
                "decode parity: structured call");
        checks.check(
                callsB.equals(oldDetectorCalls(tok, replyB)),
                "decode parity: same calls as the retired Lfm2ToolCallDetector");

        // reply C: think span, then a call
        IntSequence replyC =
                IntSequence.of(thinkOpen)
                        .concat(tok.encode("need the search tool"))
                        .concat(IntSequence.of(thinkClose))
                        .concat(tok.encode("\n\n"))
                        .concat(IntSequence.of(tcStart))
                        .concat(tok.encode("[web_search(q='rivers', top_k=3)]"))
                        .concat(IntSequence.of(tcEnd));
        Conversation convC =
                new Conversation(
                        List.of(Message.user("search rivers")),
                        List.of(Lfm2ToolOracle.SEARCH),
                        true,
                        "");
        Message msgC = decode(template, replyC);
        checks.check(
                law(template, convC, msgC, replyC, close, scaffold),
                "round-trip law: think + tool-call reply splices verbatim");
        checks.check(
                toolCalls(msgC).equals(oldDetectorCalls(tok, replyC)),
                "decode parity: think + call, same calls as retired detector");

        // one-shot decode and streaming feed agree
        checks.check(
                template.decode(replyA).equals(msgA),
                "one-shot decode(reply) equals the streamed decode");

        checks.finish("Lfm2CodecLawOracle", "all laws hold");
    }

    private static Message decode(Lfm2ChatTemplate template, IntSequence reply) {
        ReplyDecoder d = template.decoder();
        reply.forEachInt(d::feed);
        d.finish();
        return d.message();
    }

    /** encode(conv + reply-message) == encode(conv) ++ reply ++ closeTurn ++ scaffold. */
    private static boolean law(
            Lfm2ChatTemplate template,
            Conversation conv,
            Message reply,
            IntSequence replyIds,
            IntSequence close,
            IntSequence scaffold) {
        IntSequence base = IntSequence.wrap(Batch.tokenIds(template.encode(conv)));
        IntSequence extended =
                IntSequence.wrap(Batch.tokenIds(template.encode(conv.append(reply))));
        IntSequence expected = base.concat(replyIds).concat(close).concat(scaffold);
        boolean equal = IntSequence.contentEquals(extended, expected);
        if (!equal) {
            System.out.println("  expected: " + expected);
            System.out.println("  actual:   " + extended);
        }
        return equal;
    }

    private static String reasoningText(Message m) {
        StringBuilder sb = new StringBuilder();
        for (Part p : m.content()) {
            if (p instanceof Part.Reasoning r) {
                for (Part inner : r.content()) {
                    if (inner instanceof Part.Text t) sb.append(t.text());
                }
            }
        }
        return sb.toString();
    }

    private static List<Part.ToolCall> toolCalls(Message m) {
        List<Part.ToolCall> calls = new ArrayList<>();
        for (Part p : m.content()) {
            if (p instanceof Part.ToolCall c) {
                // parity comparison is on name+arguments (the retired detector had no verbatim)
                calls.add(new Part.ToolCall(c.id(), c.name(), c.arguments()));
            } else if (p instanceof Part.Reasoning r) {
                for (Part inner : r.content()) {
                    if (inner instanceof Part.ToolCall c) {
                        calls.add(new Part.ToolCall(c.id(), c.name(), c.arguments()));
                    }
                }
            }
        }
        return calls;
    }

    /** The retired Lfm2ToolCallDetector, inlined verbatim as the parity reference. */
    private static List<Part.ToolCall> oldDetectorCalls(
            GgufTokenizer tokenizer, IntSequence reply) {
        int startMarker = tokenizer.requiredSpecial("<|tool_call_start|>");
        int endMarker = tokenizer.requiredSpecial("<|tool_call_end|>");
        ByteArrayOutputStream span = new ByteArrayOutputStream();
        List<Part.ToolCall> calls = new ArrayList<>();
        boolean[] inSpan = {false};
        reply.forEachInt(
                token -> {
                    if (token == startMarker) {
                        inSpan[0] = true;
                        span.reset();
                    } else if (token == endMarker) {
                        if (inSpan[0]) {
                            calls.addAll(
                                    ToolCallSyntax.parseBlock(
                                            span.toString(StandardCharsets.UTF_8)));
                            span.reset();
                            inSpan[0] = false;
                        }
                    } else if (inSpan[0]) {
                        byte[] bytes = tokenizer.decodeTokenBytes(token);
                        span.write(bytes, 0, bytes.length);
                    }
                });
        return calls;
    }
}
