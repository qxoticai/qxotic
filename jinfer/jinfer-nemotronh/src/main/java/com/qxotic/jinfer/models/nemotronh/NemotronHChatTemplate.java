package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.PromptWriter;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Nemotron-H chat framing (ChatML dialect), token-exact with the GGUF's Jinja chat_template: no
 * bos, per turn {@code <|im_start|>{role}\n{content}<|im_end|>\n}, generation prompt {@code
 * <|im_start|>assistant\n<think>\n} to reason or {@code <|im_start|>assistant\n<think></think>} to
 * answer directly. A historical assistant turn is prefixed with an empty {@code <think></think>};
 * content with both markers keeps only the text after the LAST {@code </think>} behind the empty
 * pair.
 *
 * <p>The template unconditionally renders a system turn: {@link #DEFAULT_SYSTEM} is injected when
 * the first message is not a system turn. Two shapes: plain conversations fold per turn
 * (turn-stable blocks, the cached-prompt law); tool-bearing ones render the whole flow
 * (declarations + instructions in the system turn, call turns, folded tool responses).
 */
public final class NemotronHChatTemplate implements ChatTemplate {

    public static final String DEFAULT_SYSTEM =
            "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.";

    /**
     * The format-instructions block after the declarations - ONE constant, exactly the template's
     * string: {@link PromptWriter#trusted} mints the literal {@code <tool_call>} spellings (example
     * and reminder alike) as ids, matching the render+rescan.
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

    private final Tokenizer tokenizer;
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    // the tool markers, or -1 on older vocabs without them (tools then punt / parse plainly)
    private final int toolCall; // <tool_call>
    private final int endToolCall; // </tool_call>
    private final int toolResponse; // <tool_response>
    private final int endToolResponse; // </tool_response>
    private final IntSequence seedThinking; // <think>\n
    private final IntSequence seedDirect; // <think>\n\n</think>\n\n

    public NemotronHChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        think = SpecialTokens.require(tokenizer, "<think>");
        endThink = SpecialTokens.require(tokenizer, "</think>");
        toolCall = SpecialTokens.find(tokenizer, "<tool_call>").orElse(-1);
        endToolCall = SpecialTokens.find(tokenizer, "</tool_call>").orElse(-1);
        toolResponse = SpecialTokens.find(tokenizer, "<tool_response>").orElse(-1);
        endToolResponse = SpecialTokens.find(tokenizer, "</tool_response>").orElse(-1);
        IntSequence.Builder thinking = IntSequence.newBuilder();
        thinking.add(think).addAll(tokenizer.encode("\n"));
        seedThinking = thinking.build();
        IntSequence.Builder direct = IntSequence.newBuilder();
        direct.add(think)
                .addAll(tokenizer.encode("\n\n"))
                .add(endThink)
                .addAll(tokenizer.encode("\n\n"));
        seedDirect = direct.build();
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        Objects.requireNonNull(conversation, "conversation");
        List<Message> msgs = normalize(conversation.messages());
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        if (conversation.tools().isEmpty() && plainShape(msgs)) {
            for (Message m : msgs) {
                writePlainTurn(out, m);
                out.flush();
            }
        } else {
            requireToolShapes(msgs);
            if (toolCall < 0) {
                throw new UnsupportedConversation(
                        "tools need the <tool_call> markers in the vocab");
            }
            writeToolConversation(out, conversation, msgs);
            out.flush();
        }
        IntSequence replyPrefix = conversation.thinking() ? seedThinking : seedDirect;
        writeGenerationPrompt(out, conversation.thinking());
        out.finish();

        ReplyParser parser = spans().parser();
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    /**
     * The template unconditionally renders a system turn: inject the default when absent, so every
     * caller frames identically.
     */
    private static List<Message> normalize(List<Message> msgs) {
        if (!msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM)) {
            return msgs;
        }
        List<Message> out = new ArrayList<>(msgs.size() + 1);
        out.add(new Message(Role.SYSTEM, DEFAULT_SYSTEM));
        out.addAll(msgs);
        return out;
    }

    /** {@code <|im_start|>{role}\n{content}<|im_end|>\n} - one contiguous run per turn. */
    private void writePlainTurn(PromptWriter out, Message m) {
        out.id(imStart);
        if (m.role().equals(Role.ASSISTANT)) {
            String c = m.text();
            int lastClose = c.lastIndexOf("</think>");
            out.text("assistant\n");
            if (c.contains("<think>") == (lastClose >= 0)) {
                // framed (both markers -> keep the tail after the last </think>) or plain (no
                // markers -> the whole content): the empty pair, then the text with its trailing
                // whitespace stripped (leading whitespace survives, as in the template)
                String rest =
                        (lastClose >= 0 ? c.substring(lastClose + "</think>".length()) : c)
                                .stripTrailing();
                out.id(think).id(endThink);
                if (!rest.isEmpty()) out.text(rest);
            } else {
                // "broken thought" (unpaired marker): passthrough, fully stripped
                String rest = c.strip();
                if (!rest.isEmpty()) out.text(rest);
            }
        } else {
            // user/system: role header + content is ONE contiguous run between the specials
            out.text(m.role().name() + "\n" + m.text());
        }
        out.id(imEnd).text("\n");
    }

    /** {@code <|im_start|>assistant\n<think>\n} to reason, {@code ...<think></think>} to answer. */
    private void writeGenerationPrompt(PromptWriter out, boolean thinking) {
        out.id(imStart).text("assistant\n").id(think);
        if (thinking) {
            out.text("\n");
        } else {
            out.id(endThink);
        }
    }

    /** The template's whole tool flow: declarations, call turns, folded tool responses. */
    private void writeToolConversation(
            PromptWriter out, Conversation conversation, List<Message> msgs) {
        Message sys = msgs.get(0).role().equals(Role.SYSTEM) ? msgs.get(0) : null;
        String sysText = sys != null ? sys.text() : DEFAULT_SYSTEM;
        List<Message> loop = sys != null ? msgs.subList(1, msgs.size()) : msgs;
        int lastUser = -1;
        for (int i = 0; i < loop.size(); i++) {
            if (loop.get(i).role().equals(Role.USER)) lastUser = i;
        }

        // system turn: message text, then declarations + instructions when tools are offered
        out.id(imStart).text("system\n").text(sysText);
        if (!conversation.tools().isEmpty()) {
            if (!sysText.isEmpty()) out.text("\n\n");
            out.text(NemotronToolSyntax.declarations(conversation.tools()));
            out.trusted(TOOL_INSTRUCTIONS);
        }
        out.id(imEnd).text("\n");

        for (int i = 0; i < loop.size(); i++) {
            Message m = loop.get(i);
            if (m.role().equals(Role.TOOL)) {
                if (i == 0 || !loop.get(i - 1).role().equals(Role.TOOL)) {
                    out.id(imStart).text("user\n");
                }
                out.id(toolResponse).text("\n");
                for (Content part : m.content()) {
                    // both wire shapes carry the result text: typed ToolResult (framework
                    // adapters) and plain Text (the server's lowering shape)
                    switch (part) {
                        case Content.ToolResult r -> out.text(r.text());
                        case Content.Text t -> out.text(t.text());
                        default -> {}
                    }
                }
                out.text("\n").id(endToolResponse).text("\n");
                boolean nextIsTool =
                        i + 1 < loop.size() && loop.get(i + 1).role().equals(Role.TOOL);
                if (!nextIsTool) out.id(imEnd).text("\n");
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                out.id(imStart).text(m.role().name()).text("\n").text(m.text());
                out.id(imEnd).text("\n");
                continue;
            }
            List<Content.ToolCall> calls = callsOf(m);
            out.id(imStart).text("assistant\n");
            Content.Reasoning reasoning = reasoningOf(m);
            boolean truncate = i < lastUser; // truncate_history_thinking, template default true
            if (reasoning != null && !truncate) {
                out.id(think).text("\n").text(reasoning.text()).text("\n").id(endThink);
                // the "\n" after </think> survives the template's trim only when text follows
                // (the call branch's own '\n' re-adds it before <tool_call> blocks)
                String tail = m.text().stripTrailing();
                if (!tail.isEmpty()) out.text("\n").text(tail);
            } else {
                // truncated (or no reasoning): the empty pair then the text - trailing-trimmed
                // when kept as-is, fully stripped through the template's truncation split
                out.id(think).id(endThink);
                String tail = truncate ? m.text().strip() : m.text().stripTrailing();
                if (!tail.isEmpty()) out.text(tail);
            }
            if (!calls.isEmpty()) {
                out.text("\n"); // the call branch appends '\n' after the content
                for (Content.ToolCall call : calls) {
                    out.id(toolCall).text(NemotronToolSyntax.call(call)).id(endToolCall);
                    out.text("\n");
                }
            }
            out.id(imEnd).text("\n");
        }
    }

    private static Content.Reasoning reasoningOf(Message m) {
        for (Content part : m.content()) {
            if (part instanceof Content.Reasoning reasoning) return reasoning;
        }
        return null;
    }

    private static List<Content.ToolCall> callsOf(Message m) {
        return m.content().stream()
                .filter(Content.ToolCall.class::isInstance)
                .map(Content.ToolCall.class::cast)
                .toList();
    }

    private static boolean plainShape(List<Message> msgs) {
        for (Message m : msgs) {
            for (Content part : m.content()) {
                if (!(part instanceof Content.Text)) return false;
            }
        }
        return true;
    }

    /** The part shapes the tool flow frames byte-exactly; anything else is rejected. */
    private static void requireToolShapes(List<Message> msgs) {
        for (Message m : msgs) {
            boolean assistant = m.role().equals(Role.ASSISTANT);
            boolean tool = m.role().equals(Role.TOOL);
            for (Content part : m.content()) {
                boolean ok =
                        part instanceof Content.Text
                                || (assistant
                                        && (part instanceof Content.ToolCall
                                                || part instanceof Content.Reasoning))
                                || (tool && part instanceof Content.ToolResult);
                if (!ok)
                    throw new UnsupportedConversation(
                            m.role().name() + " turn: " + part.getClass().getSimpleName());
            }
        }
    }

    @Override
    public ReplyParser parser(Tokenizer tokenizer) {
        return spans().parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(
            String contentGbnf, List<Tool> callableTools) {
        return Optional.of(spans().constrainedAuto(contentGbnf, !callableTools.isEmpty()));
    }

    /** Forced calls: the header carries an OFFERED name, the arguments stay the model's own. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(spans().forcedCall(callableTools, tool -> "\n<function=" + tool.name()));
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
}
