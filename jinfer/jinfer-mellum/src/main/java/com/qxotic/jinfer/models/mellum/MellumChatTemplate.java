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
 * verbatim, never trimmed), generation prompt {@code <|im_start|>assistant\n}. The model answers
 * directly - there is no thinking mode - and the template keeps only what follows the last {@code
 * </think>} of an echoed assistant turn.
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
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private ReplyLanguage.Spans spans; // the family's derived faces, markers written once

    MellumChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
    }

    @Override
    public ThinkingPolicy thinkingPolicy() {
        return ThinkingPolicy.NONE;
    }

    @Override
    public int defaultMaxReasoningTokens(int maxOutputTokens) {
        return -1;
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
                writeAssistant(out, m);
                out.flush();
            } else {
                writeTurn(out, m.role().name(), text(m));
                out.flush();
            }
        }
        out.id(imStart).text("assistant\n");
        out.finish();
        return new ReplyState(IntSequence.empty(), spans().parser());
    }

    /** {@code <|im_start|>{role}\n{content}<|im_end|>\n} - one contiguous run per turn. */
    private void writeTurn(PromptWriter out, String role, String content) {
        out.id(imStart).text(role + "\n" + content).id(imEnd).text("\n");
    }

    /**
     * The assistant turn: the text after its last {@code </think>} (leading newlines dropped, as
     * the template's {@code split('</think>')[-1].lstrip('\n')}), then one envelope per call - a
     * newline before every call except a first one after empty content.
     */
    private void writeAssistant(PromptWriter out, Message m) {
        String content = afterThinking(text(m));
        out.id(imStart).text("assistant\n" + content);
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

    /** The turn's text and tool-result parts, concatenated; reasoning is never echoed. */
    private static String text(Message m) {
        StringBuilder text = new StringBuilder();
        for (Content part : m.content()) {
            if (part instanceof Content.Text value) text.append(value.text());
            else if (part instanceof Content.ToolResult value) text.append(value.text());
        }
        return text.toString();
    }

    /** Python {@code content.split('</think>')[-1].lstrip('\n')}. */
    private static String afterThinking(String content) {
        int at = content.lastIndexOf("</think>");
        String tail = at < 0 ? content : content.substring(at + "</think>".length());
        int i = 0;
        while (i < tail.length() && tail.charAt(i) == '\n') i++;
        return tail.substring(i);
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
        return spans().parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String grammar) {
        return Optional.of(spans().constrained(grammar));
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String grammar, boolean calls) {
        return Optional.of(spans().constrained(grammar, calls));
    }

    /** Forced calls: the envelope carries an OFFERED name, the schema binds the arguments. */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(
                ReplyLanguage.Selection.of(
                        JsonEnvelopeReplies.forced(callableTools, "<|im_end|>"), tokenizer));
    }

    private ReplyLanguage.Spans spans() {
        if (spans == null) {
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
        return spans;
    }
}
