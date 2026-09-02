package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.Batch;
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
     * What a checkpoint's template does that the codec must mirror, read off its source. Nemotron
     * Cascade 2 injects {@link #DEFAULT_SYSTEM} when no system turn is given, frames typed
     * reasoning as {@code <think>\n{R}\n</think>\n} and trims a truncated call turn's tail;
     * Nemotron 3.5 renders an EMPTY system turn, frames {@code <think>\n{R}</think>} and keeps the
     * tail as it is.
     */
    public record Dialect(
            String defaultSystem, boolean reasoningNewlines, boolean trimsTruncatedCallTail) {
        public static final Dialect CASCADE = new Dialect(DEFAULT_SYSTEM, true, true);
        public static final Dialect LIGHTNING = new Dialect("", false, false);

        public static Dialect of(String templateSource) {
            return templateSource.contains("You are a helpful and harmless assistant")
                    ? CASCADE
                    : LIGHTNING;
        }
    }

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
    private final Dialect dialect;
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
        this(tokenizer, Dialect.CASCADE);
    }

    public NemotronHChatTemplate(Tokenizer tokenizer, Dialect dialect) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.dialect = Objects.requireNonNull(dialect, "dialect");
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
        // a tool-result turn renders as the template's folded user turn whether or not tools
        // are offered on THIS request (the server's lowering shape carries history without them)
        boolean toolTurns = msgs.stream().anyMatch(m -> m.role().equals(Role.TOOL));
        if (conversation.tools().isEmpty() && plainShape(msgs) && !toolTurns) {
            int lastUser = lastUser(msgs);
            for (int i = 0; i < msgs.size(); i++) {
                writePlainTurn(out, msgs.get(i), i < lastUser);
                out.flush();
            }
        } else {
            requireToolShapes(msgs);
            if (toolCall < 0 || toolResponse < 0) {
                throw new UnsupportedConversation(
                        "tools need the <tool_call> and <tool_response> markers in the vocab");
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
    private List<Message> normalize(List<Message> msgs) {
        if (!msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM)) {
            return msgs;
        }
        List<Message> out = new ArrayList<>(msgs.size() + 1);
        out.add(new Message(Role.SYSTEM, dialect.defaultSystem()));
        out.addAll(msgs);
        return out;
    }

    private static int lastUser(List<Message> msgs) {
        int last = -1;
        for (int i = 0; i < msgs.size(); i++) if (msgs.get(i).role().equals(Role.USER)) last = i;
        return last;
    }

    /**
     * {@code <|im_start|>{role}\n{content}<|im_end|>\n} - one contiguous run per turn. An assistant
     * turn before the last user is truncated to the empty pair plus its answer; one after it keeps
     * its content (trimmed), inline think markers minted as the ids.
     */
    private void writePlainTurn(PromptWriter out, Message m, boolean truncate) {
        out.id(imStart);
        if (m.role().equals(Role.ASSISTANT)) {
            out.text("assistant\n");
            writeMarked(out, assistantBody(withPair(m.text()), truncate, false));
        } else {
            // user/system: role header + content is ONE contiguous run between the specials
            out.text(m.role().name() + "\n" + m.text());
        }
        out.id(imEnd).text("\n");
    }

    /** The template's first move on an assistant turn: no marker at all gets the empty pair. */
    private static String withPair(String text) {
        return text.contains("<think>") || text.contains("</think>")
                ? text
                : "<think></think>" + text;
    }

    /**
     * The template's assistant content rule. Kept (at or after the last user): the content,
     * trimmed. Truncated before it: the empty pair plus what follows the last close; the call
     * branch also drops a dangling open's tail and trims per dialect, the plain branch trims the
     * whole and leaves a broken thought as it is.
     */
    private String assistantBody(String content, boolean truncate, boolean callBranch) {
        if (!truncate) return content.strip();
        int close = content.lastIndexOf("</think>");
        if (callBranch) {
            String tail =
                    close >= 0
                            ? content.substring(close + "</think>".length())
                            : content.contains("<think>")
                                    ? content.substring(0, content.indexOf("<think>"))
                                    : content;
            return "<think></think>" + (dialect.trimsTruncatedCallTail() ? tail.strip() : tail);
        }
        boolean paired = content.contains("<think>") && close >= 0;
        return (paired
                        ? "<think></think>" + content.substring(close + "</think>".length())
                        : content)
                .strip();
    }

    /** Inline think markers are the ids, as the template path mints them; the rest is text. */
    private void writeMarked(PromptWriter out, String text) {
        int at = 0;
        while (at < text.length()) {
            int open = text.indexOf("<think>", at), close = text.indexOf("</think>", at);
            int next = open < 0 ? close : close < 0 ? open : Math.min(open, close);
            if (next < 0) break;
            if (next > at) out.text(text.substring(at, next));
            boolean isOpen = next == open;
            out.id(isOpen ? think : endThink);
            at = next + (isOpen ? "<think>".length() : "</think>".length());
        }
        if (at < text.length()) out.text(text.substring(at));
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
        String sysText = sys != null ? sys.text() : dialect.defaultSystem();
        List<Message> loop = sys != null ? msgs.subList(1, msgs.size()) : msgs;
        int lastUser = lastUser(loop);

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
            // typed reasoning is framed into the content first (the template's own move), then
            // the content goes through the same keep-or-truncate rule as inline markers
            String nl = dialect.reasoningNewlines() ? "\n" : "";
            String content =
                    reasoning != null
                            ? "<think>\n" + reasoning.text() + nl + "</think>" + nl + m.text()
                            : withPair(m.text());
            writeMarked(out, assistantBody(content, truncate, !calls.isEmpty()));
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
    public Optional<ReplyLanguage.Selection> constrainedReply(String contentGbnf) {
        return Optional.of(spans().constrained(contentGbnf));
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
