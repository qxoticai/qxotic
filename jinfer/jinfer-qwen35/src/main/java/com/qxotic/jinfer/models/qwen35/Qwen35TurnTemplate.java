package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Hand-written Qwen3.5 chat framing (ChatML dialect), token-exact with the GGUF's Jinja
 * chat_template over plain conversations.
 *
 * <p>Layout: NO bos (Qwen3.5 has none; {@link #conversationStart} is empty), per turn {@code
 * <|im_start|>{role}\n{content|trim}<|im_end|>\n}. Matching the template, every turn's content is
 * trimmed, and a historical assistant turn keeps only the text after its last {@code </think>}
 * (leading newlines stripped) - the template's frozen middle-turn form; a trailing assistant turn
 * after the final user query renders differently (thinking kept) and is out of scope for
 * turn-stable encoding, as in the other curated templates.
 *
 * <p>Generation prompt: {@code <|im_start|>assistant\n} then the thinking scaffold - {@code
 * <think>\n} to reason, or the pre-closed {@code <think>\n\n</think>\n\n} to answer directly. The
 * 2B template defaults to NON-thinking ({@code enable_thinking} must be defined and true to
 * reason); note the 35B-A3B template INVERTS that default (thinking unless {@code enable_thinking}
 * is defined and false) - the scaffolds themselves are identical, only the default flag differs, so
 * this template serves both.
 *
 * <p>Each text run between specials is ONE contiguous plain {@link Tokenizer#encode}; conversation
 * content never goes through special-aware encoding, so text cannot mint control tokens ({@code
 * <think>}/{@code </think>} in the scaffold are emitted as trusted ids).
 */
public final class Qwen35TurnTemplate implements TurnTemplate {

    private final Tokenizer tokenizer;
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    private final IntSequence newline; // encode("\n"), constant
    private final List<Batch> genThinking, genDirect; // generation prompts, encoded once
    private final List<Batch> closeTurn; // <|im_end|>\n, constant
    private final TokenRuns proto; // compiled spelling table, forked per encode

    public Qwen35TurnTemplate(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        this.imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        this.think = SpecialTokens.require(tokenizer, "<think>");
        this.endThink = SpecialTokens.require(tokenizer, "</think>");
        this.newline = tokenizer.encode("\n");
        // <|im_start|>assistant\n<think>\n            (reasoning)
        // <|im_start|>assistant\n<think>\n\n</think>\n\n   (direct answer)
        IntSequence head = IntSequence.of(imStart).concat(tokenizer.encode("assistant\n"));
        IntSequence thinking = head.concat(IntSequence.of(think)).concat(tokenizer.encode("\n"));
        this.genThinking = List.of(Batch.prefill(thinking.toArray()));
        IntSequence direct =
                head.concat(IntSequence.of(think))
                        .concat(tokenizer.encode("\n\n"))
                        .concat(IntSequence.of(endThink))
                        .concat(tokenizer.encode("\n\n"));
        this.genDirect = List.of(Batch.prefill(direct.toArray()));
        IntSequence close = IntSequence.of(imEnd).concat(newline);
        this.closeTurn = List.of(Batch.prefill(close.toArray()));
        this.proto = new TokenRuns(tokenizer);
    }

    /** Qwen3.5 emits no bos and no fixed preamble. */
    @Override
    public List<Batch> conversationStart() {
        return List.of();
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        String content = message.textOnly().strip(); // template: content|trim
        if (message.role().equals(Role.ASSISTANT)) content = stripThinking(content);
        IntSequence ids =
                IntSequence.of(imStart)
                        .concat(tokenizer.encode(message.role().name() + "\n" + content))
                        .concat(IntSequence.of(imEnd))
                        .concat(newline);
        return List.of(Batch.prefill(ids.toArray()));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return thinking ? genThinking : genDirect;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    /**
     * The template keeps only the text after the last {@code </think>}, leading newlines stripped:
     * {@code content.split('</think>')[-1].lstrip('\n')}.
     */
    private static String stripThinking(String content) {
        int at = content.lastIndexOf("</think>");
        if (at < 0) return content;
        String tail = content.substring(at + "</think>".length());
        int i = 0;
        while (i < tail.length() && tail.charAt(i) == '\n') i++;
        return tail.substring(i);
    }

    /**
     * The format-instructions block after the declarations - ONE constant, exactly the template's
     * string; its literal {@code <tool_call>} spellings are emitted as trusted ids, matching the
     * render+rescan. (Byte-identical to Nemotron's - the two families share the call wire.)
     */
    static final String TOOL_INSTRUCTIONS =
            "\n\n"
                + "If you choose to call a function ONLY reply in the following format with NO"
                + " suffix:\n\n"
                + "<tool_call>\n"
                + "<function=example_function_name>\n"
                + "<parameter=example_parameter_1>\n"
                + "value_1\n"
                + "</parameter>\n"
                + "<parameter=example_parameter_2>\n"
                + "This is the value for the second parameter\n"
                + "that can span\n"
                + "multiple lines\n"
                + "</parameter>\n"
                + "</function>\n"
                + "</tool_call>\n\n"
                + "<IMPORTANT>\n"
                + "Reminder:\n"
                + "- Function calls MUST follow the specified format: an inner"
                + " <function=...></function> block must be nested within <tool_call></tool_call>"
                + " XML tags\n"
                + "- Required parameters MUST be specified\n"
                + "- You may provide optional reasoning for your function call in natural language"
                + " BEFORE the function call, but NOT after\n"
                + "- If there is no function call available, answer the question like normal with"
                + " your current knowledge and do not tell the user about function calls\n"
                + "</IMPORTANT>";

    /**
     * The codec face, tools included - the template's whole-conversation flow. Plain conversations
     * keep the oracle-validated per-turn fold; tool-bearing ones render natively, matching the GGUF
     * template: declarations ({@code # Tools ... <tools>} with each tool's {@code tojson}) plus the
     * format instructions inside the system turn; assistant call turns as content then one {@code
     * <tool_call>} XML function block per call; tool results folded into a single {@code user} turn
     * of {@code <tool_response>} blocks (BOTH wire shapes render - typed {@link Part.ToolResult}
     * and the server's plain {@link Part.Text} lowering); and the template's thinking policy -
     * reasoning is kept only on assistant turns AFTER the last real user query (a user turn that is
     * not a {@code <tool_response>} wrapper), and dropped entirely on historical turns (no empty
     * pair - unlike Nemotron).
     */
    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        if (conversation.tools().isEmpty() && plainShape(msgs)) {
            return TurnTemplate.super.encode(conversation);
        }
        TurnTemplate.requireToolShapes(msgs);
        SpecialTokens.require(tokenizer, "<tool_call>"); // the call wire needs the markers
        SpecialTokens.require(tokenizer, "</tool_call>");

        Message sys = Message.leadingSystem(msgs);
        TokenRuns runs = proto.fresh();
        if (!conversation.tools().isEmpty()) {
            runs.id(imStart).text("system\n");
            runs.text("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for (Tool tool : conversation.tools()) {
                runs.text("\n").text(ToolCallSyntax.jinjaJson(JsonCodec.parse(tool.rawJson())));
            }
            runs.text("\n</tools>");
            runs.trusted(TOOL_INSTRUCTIONS);
            if (sys != null) {
                String sysText = sys.textOnly().strip();
                if (!sysText.isEmpty()) runs.text("\n\n").text(sysText);
            }
            runs.id(imEnd).text("\n");
        } else if (sys != null) {
            runs.id(imStart).text("system\n").text(sys.textOnly().strip()).id(imEnd).text("\n");
        }

        // the template's walk-back: the last USER turn that is not a <tool_response> wrapper;
        // assistant turns AFTER it keep their reasoning, all earlier ones drop it entirely
        int lastQuery = -1;
        for (int i = msgs.size() - 1; i >= 0; i--) {
            Message m = msgs.get(i);
            if (!m.role().equals(Role.USER)) continue;
            String c = m.textOnly().strip();
            if (c.startsWith("<tool_response>") && c.endsWith("</tool_response>")) continue;
            lastQuery = i;
            break;
        }

        for (int i = 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.SYSTEM)) continue; // rendered above (template: first only)
            if (m.role().equals(Role.TOOL)) {
                if (i == 0 || !msgs.get(i - 1).role().equals(Role.TOOL)) {
                    runs.id(imStart).text("user");
                }
                runs.trusted("\n<tool_response>\n");
                StringBuilder result = new StringBuilder();
                for (Part p : m.content()) {
                    switch (p) {
                        case Part.ToolResult r -> result.append(r.text());
                        case Part.Text t -> result.append(t.text());
                        default -> {}
                    }
                }
                runs.text(result.toString().strip());
                runs.trusted("\n</tool_response>");
                boolean nextIsTool =
                        i + 1 < msgs.size() && msgs.get(i + 1).role().equals(Role.TOOL);
                if (!nextIsTool) runs.id(imEnd).text("\n");
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                runs.id(imStart).text(m.role().name()).text("\n").text(m.textOnly().strip());
                runs.id(imEnd).text("\n");
                continue;
            }
            String content = m.text().strip();
            Part.Reasoning reasoning = m.reasoning();
            String reasoningText;
            if (reasoning != null) {
                reasoningText = reasoning.text().strip();
            } else {
                int at = content.lastIndexOf("</think>");
                if (at >= 0) {
                    String head = content.substring(0, at);
                    int open = head.lastIndexOf("<think>");
                    reasoningText =
                            stripNl(open >= 0 ? head.substring(open + "<think>".length()) : head)
                                    .strip();
                } else {
                    reasoningText = "";
                }
                content =
                        stripLeadingNl(
                                at >= 0 ? content.substring(at + "</think>".length()) : content);
            }
            runs.id(imStart).text("assistant\n");
            if (i > lastQuery) {
                runs.id(think)
                        .text("\n")
                        .text(reasoningText)
                        .text("\n")
                        .id(endThink)
                        .text("\n\n")
                        .text(content);
            } else {
                runs.text(content);
            }
            List<Part.ToolCall> calls =
                    m.content().stream()
                            .filter(p -> p instanceof Part.ToolCall)
                            .map(p -> (Part.ToolCall) p)
                            .toList();
            for (int c = 0; c < calls.size(); c++) {
                Part.ToolCall call = calls.get(c);
                if (c == 0) {
                    runs.text(content.isEmpty() ? "" : "\n\n");
                } else {
                    runs.text("\n");
                }
                runs.trusted("<tool_call>\n");
                runs.text("<function=" + call.name() + ">\n");
                for (Map.Entry<String, Object> arg : call.arguments().entrySet()) {
                    runs.text("<parameter=" + arg.getKey() + ">\n");
                    runs.text(ToolCallSyntax.jinjaValue(arg.getValue()));
                    runs.text("\n</parameter>\n");
                }
                runs.text("</function>\n");
                runs.trusted("</tool_call>");
            }
            runs.id(imEnd).text("\n");
        }
        List<Batch> out = new ArrayList<>();
        out.add(runs.batch());
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }

    private static boolean plainShape(List<Message> msgs) {
        for (Message m : msgs) {
            for (Part p : m.content()) {
                if (!(p instanceof Part.Text)) return false;
            }
        }
        return true;
    }

    /** Python {@code lstrip('\n')}. */
    private static String stripLeadingNl(String s) {
        int i = 0;
        while (i < s.length() && s.charAt(i) == '\n') i++;
        return s.substring(i);
    }

    /** The template's {@code rstrip('\n')} + {@code lstrip('\n')} around the think split. */
    private static String stripNl(String s) {
        int end = s.length();
        while (end > 0 && s.charAt(end - 1) == '\n') end--;
        return stripLeadingNl(s.substring(0, end));
    }

    private ReplyLanguage.Spans spans; // the family's derived faces, markers written once

    private ReplyLanguage.Spans spans() {
        if (spans == null) {
            spans =
                    new ReplyLanguage.Spans(
                            "<think>",
                            "</think>",
                            "<tool_call>",
                            "</tool_call>",
                            ToolCallSyntax::parseFunctionXml,
                            ReplyLanguage.mark("<|im_end|>"),
                            tokenizer);
        }
        return spans;
    }

    /** The reply-language walk over {@code think? (content | tool_call-span)* im_end?}. */
    @Override
    public ReplyParser parser() {
        return spans().parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedAuto(String contentGbnf) {
        return Optional.of(spans().constrainedAuto(contentGbnf));
    }

    /** Forced calls seed {@code <tool_call>}; the pin below holds the name. */
    @Override
    public int[] callSeed() {
        return new int[] {SpecialTokens.require(tokenizer, "<tool_call>")};
    }

    /** The {@code <function=} header this family emits after the marker. */
    @Override
    public Optional<String> callPrefix() {
        return Optional.of("\n<function=");
    }

    /** The generation prompt opens the think span (or its closed pair): pre-feed it. */
    @Override
    public int[] replySeed(boolean thinking) {
        return TurnTemplate.reasonSeed(tokenizer, thinking);
    }
}
