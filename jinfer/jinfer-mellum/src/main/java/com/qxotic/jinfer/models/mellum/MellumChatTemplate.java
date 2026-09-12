package com.qxotic.jinfer.models.mellum;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonEnvelopeReplies;
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
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Mellum 2 chat framing (ChatML with Hermes-style JSON tool calls), token-exact with the GGUF's
 * Jinja chat_template: no bos, per turn {@code <|im_start|>{role}\n{content}<|im_end|>\n} (content
 * verbatim, never trimmed), generation prompt {@code <|im_start|>assistant\n}.
 *
 * <p>Two checkpoints share the frame. Instruct answers directly: no thinking mode, and only what
 * follows the last {@code </think>} of an echoed assistant turn is kept. Thinking opens its own
 * {@code <think>} span, so thinking on adds nothing to the prompt and thinking off pre-closes the
 * span ({@code <think>\n\n</think>\n\n}); echoed reasoning is kept only on assistant turns after
 * the last real user query, as the template's {@code last_query_index} walk-back does.
 *
 * <p>With tools the system turn carries the declarations ({@code # Tools ... <tools>} with each
 * tool's {@code tojson}) and the call-format instructions; assistant call turns render their
 * content then one {@code <tool_call>} JSON envelope per call; consecutive tool results fold into
 * one {@code user} turn of {@code <tool_response>} blocks.
 */
final class MellumChatTemplate implements ChatTemplate {

    /** The template's declarations preamble, after the optional system text. */
    static final String TOOLS_HEAD =
            "# Tools\n\n"
                    + "You may call one or more functions to assist with the user query.\n\n"
                    + "You are provided with function signatures within <tools></tools> XML"
                    + " tags:\n<tools>";

    /** The call-format instructions, ONE constant, exactly the template's string. */
    static final String TOOLS_TAIL =
            "\n</tools>\n\n"
                    + "For each function call, return a json object with function name and"
                    + " arguments within <tool_call></tool_call> XML tags:\n"
                    + "<tool_call>\n"
                    + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                    + "</tool_call>";

    private final Tokenizer tokenizer;
    private final boolean reasons; // the Thinking checkpoint, whose template has enable_thinking
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final IntSequence closedThink; // <think>\n\n</think>\n\n, the thinking-off prefix
    private final ReplyLanguage.Spans spans; // the family's derived faces, markers written once

    /**
     * @param reasons whether this is the Thinking checkpoint (its template has enable_thinking)
     */
    MellumChatTemplate(Tokenizer tokenizer, boolean reasons) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.reasons = reasons;
        imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        closedThink =
                reasons
                        ? IntSequence.newBuilder()
                                .add(SpecialTokens.require(tokenizer, "<think>"))
                                .addAll(tokenizer.encode("\n\n"))
                                .add(SpecialTokens.require(tokenizer, "</think>"))
                                .addAll(tokenizer.encode("\n\n"))
                                .build()
                        : IntSequence.empty();
        spans =
                new ReplyLanguage.Spans(
                        "<think>",
                        "</think>",
                        "<tool_call>",
                        "</tool_call>",
                        ToolCallSyntax::parseBlock,
                        ReplyLanguage.mark("<|im_end|>"),
                        tokenizer);
    }

    @Override
    public ThinkingPolicy thinkingPolicy() {
        return reasons ? ThinkingPolicy.OPTIONAL : ThinkingPolicy.NONE;
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        Objects.requireNonNull(conversation, "conversation");
        List<Message> msgs = conversation.messages();
        if (msgs.isEmpty())
            throw new UnsupportedConversation("Mellum requires at least one message");
        requireShapes(msgs);
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        Message system = msgs.getFirst().role().equals(Role.SYSTEM) ? msgs.getFirst() : null;
        if (!conversation.tools().isEmpty()) {
            SpecialTokens.require(tokenizer, "<tool_call>");
            SpecialTokens.require(tokenizer, "</tool_call>");
            out.id(imStart).text("system\n");
            if (system != null) out.text(text(system)).text("\n\n");
            out.text(TOOLS_HEAD);
            for (Tool tool : conversation.tools())
                out.text("\n").text(ToolCallSyntax.jinjaJson(tool.definition()));
            out.trusted(TOOLS_TAIL);
            out.id(imEnd).text("\n").flush();
        } else if (system != null) {
            writeTurn(out, "system", text(system));
            out.flush();
        }
        int lastQuery = lastQuery(msgs);
        for (int i = 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m == system) continue; // rendered above; the template skips it in its loop
            if (m.role().equals(Role.TOOL)) {
                boolean opens = i == 0 || !msgs.get(i - 1).role().equals(Role.TOOL);
                boolean closes = i + 1 == msgs.size() || !msgs.get(i + 1).role().equals(Role.TOOL);
                if (opens) out.id(imStart).text("user");
                out.trusted("\n<tool_response>\n").text(text(m)).trusted("\n</tool_response>");
                if (closes) out.id(imEnd).text("\n").flush();
            } else if (m.role().equals(Role.ASSISTANT)) {
                writeAssistant(out, m, reasons && i > lastQuery, i == msgs.size() - 1);
                out.flush();
            } else {
                writeTurn(out, m.role().name(), text(m));
                out.flush();
            }
        }
        out.id(imStart).text("assistant\n");
        IntSequence replyPrefix = conversation.thinking() ? IntSequence.empty() : closedThink;
        out.verbatim(replyPrefix);
        out.finish();
        ReplyParser parser = spans.parser();
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    /**
     * The template's walk-back: the last user turn that is not a {@code <tool_response>} wrapper,
     * or the last message when there is none.
     */
    private static int lastQuery(List<Message> msgs) {
        for (int i = msgs.size() - 1; i >= 0; i--) {
            Message m = msgs.get(i);
            if (!m.role().equals(Role.USER)) continue;
            String content = text(m);
            if (content.startsWith("<tool_response>") && content.endsWith("</tool_response>"))
                continue;
            return i;
        }
        return msgs.size() - 1;
    }

    /** {@code <|im_start|>{role}\n{content}<|im_end|>\n} - one contiguous run per turn. */
    private void writeTurn(PromptWriter out, String role, String content) {
        out.id(imStart).text(role + "\n" + content).id(imEnd).text("\n");
    }

    /**
     * The assistant turn. Its reasoning (a {@link Content.Reasoning} part, or the {@code <think>}
     * span inside its text) is split off the way the template does; {@code echoReasoning} turns
     * (the Thinking checkpoint after the last query) render it as {@code
     * <think>\n...\n</think>\n\n} when there is any, or when the turn is the prompt's last message.
     * Then one envelope per call, a newline before every call except a first one after empty
     * content.
     */
    private void writeAssistant(PromptWriter out, Message m, boolean echoReasoning, boolean last) {
        String raw = textWithReasoning(m);
        int close = raw.lastIndexOf("</think>");
        String content = stripLeading(close < 0 ? raw : raw.substring(close + "</think>".length()));
        String reasoning = "";
        if (close >= 0) {
            String head = raw.substring(0, raw.indexOf("</think>"));
            int open = head.lastIndexOf("<think>");
            reasoning = stripBoth(open < 0 ? head : head.substring(open + "<think>".length()));
        }
        out.id(imStart).text("assistant\n");
        if (echoReasoning && (last || !reasoning.isEmpty())) {
            out.trusted("<think>\n").text(reasoning).trusted("\n</think>\n\n");
        }
        out.text(content);
        boolean first = true;
        for (Content part : m.content()) {
            if (!(part instanceof Content.ToolCall call)) continue;
            if (!first || !content.isEmpty()) out.text("\n");
            out.trusted("<tool_call>\n");
            out.text(
                    "{\"name\": \""
                            + call.name()
                            + "\", \"arguments\": "
                            + ToolCallSyntax.jinjaJson(call.arguments())
                            + "}\n");
            out.trusted("</tool_call>");
            first = false;
        }
        out.id(imEnd).text("\n");
    }

    /** The turn's text and tool-result parts, concatenated. */
    private static String text(Message m) {
        StringBuilder text = new StringBuilder();
        for (Content part : m.content()) {
            if (part instanceof Content.Text value) text.append(value.text());
            else if (part instanceof Content.ToolResult value) text.append(value.text());
        }
        return text.toString();
    }

    /**
     * As {@link #text}, with reasoning parts wrapped in think markers - the whole-render's view.
     */
    private static String textWithReasoning(Message m) {
        StringBuilder text = new StringBuilder();
        for (Content part : m.content()) {
            if (part instanceof Content.Text value) text.append(value.text());
            else if (part instanceof Content.Reasoning value)
                text.append("<think>").append(value.text()).append("</think>");
        }
        return text.toString();
    }

    /** Python {@code s.lstrip('\n')}. */
    private static String stripLeading(String s) {
        int i = 0;
        while (i < s.length() && s.charAt(i) == '\n') i++;
        return s.substring(i);
    }

    /** Python {@code s.strip('\n')}. */
    private static String stripBoth(String s) {
        int end = s.length();
        while (end > 0 && s.charAt(end - 1) == '\n') end--;
        return stripLeading(s.substring(0, end));
    }

    /** The part shapes the template frames; anything else (media) is rejected loudly. */
    private static void requireShapes(List<Message> msgs) {
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
        return spans.parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String grammar) {
        return Optional.of(spans.constrained(grammar));
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String grammar, boolean calls) {
        return Optional.of(spans.constrained(grammar, calls));
    }

    /** Forced calls: the envelope carries an OFFERED name, the schema binds the arguments. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(
                ReplyLanguage.Selection.of(
                        JsonEnvelopeReplies.forced(callableTools, "<|im_end|>"), tokenizer));
    }
}
