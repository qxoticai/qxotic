package com.qxotic.jinfer.models.laguna;

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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/** Token-exact native codec for Laguna XS 2.1's system/user/assistant XML format. */
public final class LagunaChatTemplate implements ChatTemplate {
    static final String DEFAULT_SYSTEM =
            "You are a helpful, conversationally-fluent assistant made by Poolside. You are here"
                    + " to be helpful to users through natural language conversations.";
    private static final String THINK_OPEN = "<think>";
    private static final String THINK_CLOSE = "</think>";
    private static final String CALL_OPEN = "<tool_call>";
    private static final String CALL_CLOSE = "</tool_call>";
    private static final String TURN_END = "</assistant>";

    private final Tokenizer tokenizer;
    private final IntSequence promptStart;
    private final int thinkOpen;
    private final int thinkClose;
    private final ReplyLanguage.Spans spans;

    public LagunaChatTemplate(Tokenizer tokenizer, int bosToken) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        promptStart = IntSequence.of(bosToken);
        thinkOpen = SpecialTokens.require(tokenizer, THINK_OPEN);
        thinkClose = SpecialTokens.require(tokenizer, THINK_CLOSE);
        spans =
                new ReplyLanguage.Spans(
                        THINK_OPEN,
                        THINK_CLOSE,
                        CALL_OPEN,
                        CALL_CLOSE,
                        ToolCallSyntax::parseTaggedXml,
                        ReplyLanguage.mark(TURN_END),
                        tokenizer);
    }

    @Override
    public IntSequence promptStart() {
        return promptStart;
    }

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        Objects.requireNonNull(conversation, "conversation");
        List<Message> messages = conversation.messages();
        for (Message message : messages) requireSupported(message);

        int first = 0;
        String system = DEFAULT_SYSTEM;
        if (!messages.isEmpty() && messages.getFirst().role().equals(Role.SYSTEM)) {
            system = text(messages.getFirst());
            first = 1;
        }
        StringBuilder prompt = new StringBuilder();
        writeHeader(prompt, system, conversation.tools(), conversation.thinking());

        for (int i = first; i < messages.size(); i++) {
            Message message = messages.get(i);
            switch (message.role().name()) {
                case "user" -> prompt.append("<user>").append(text(message)).append("</user>\n");
                case "system" ->
                        prompt.append("<system>").append(text(message)).append("</system>\n");
                case "assistant" -> writeAssistant(prompt, message, conversation.thinking());
                case "tool" ->
                        prompt.append("<tool_response>")
                                .append(text(message))
                                .append("</tool_response>\n");
                default -> throw new UnsupportedConversation("role " + message.role().name());
            }
        }

        prompt.append("<assistant>").append(conversation.thinking() ? THINK_OPEN : THINK_CLOSE);
        IntSequence replyPrefix = IntSequence.of(conversation.thinking() ? thinkOpen : thinkClose);
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        out.verbatim(promptStart).verbatim(SpecialTokens.encode(tokenizer, prompt.toString()));
        out.finish();
        ReplyParser parser = spans.parser();
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    private static void writeHeader(
            StringBuilder out, String system, List<Tool> tools, boolean thinking) {
        if (system.isBlank() && tools.isEmpty() && !thinking) return;
        out.append("<system>");
        if (!system.isBlank()) out.append(system.stripTrailing());
        if (!system.isBlank() && !tools.isEmpty()) out.append("\n\n");
        if (!tools.isEmpty()) {
            out.append(
                    "### Tools\n\n"
                            + "You may call functions to assist with the user query.\n"
                            + "All available function signatures are listed below:\n");
            out.append("<available_tools>\n");
            for (Tool tool : tools)
                out.append(ToolCallSyntax.jinjaJson(toolEnvelope(tool))).append('\n');
            out.append("</available_tools>");
        }
        out.append("</system>\n");
    }

    private static void writeAssistant(StringBuilder out, Message message, boolean thinking) {
        out.append("<assistant>");
        if (thinking) {
            out.append(THINK_OPEN).append(reasoning(message)).append(THINK_CLOSE);
        } else {
            out.append(THINK_CLOSE);
        }
        out.append(visible(message));
        for (Content part : message.content())
            if (part instanceof Content.ToolCall call) writeCall(out, call);
        out.append("</assistant>\n");
    }

    private static void writeCall(StringBuilder out, Content.ToolCall call) {
        out.append(CALL_OPEN).append(call.name());
        for (Map.Entry<String, Object> argument : call.arguments().entrySet()) {
            out.append("<arg_key>")
                    .append(argument.getKey())
                    .append("</arg_key><arg_value>")
                    .append(
                            argument.getValue() instanceof String value
                                    ? value
                                    : ToolCallSyntax.jinjaJson(argument.getValue()))
                    .append("</arg_value>");
        }
        out.append(CALL_CLOSE);
    }

    private static Map<String, Object> toolEnvelope(Tool tool) {
        if (tool.definition().containsKey("function")) return tool.definition();
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("type", "function");
        envelope.put("function", tool.definition());
        return envelope;
    }

    private static String visible(Message message) {
        StringBuilder text = new StringBuilder();
        for (Content part : message.content())
            if (part instanceof Content.Text value) text.append(value.text());
        return text.toString();
    }

    private static String reasoning(Message message) {
        StringBuilder text = new StringBuilder();
        for (Content part : message.content())
            if (part instanceof Content.Reasoning value) {
                for (Content nested : value.content()) {
                    if (!(nested instanceof Content.Text token))
                        throw new UnsupportedConversation(
                                "reasoning contains " + nested.getClass().getSimpleName());
                    text.append(token.text());
                }
            }
        return text.toString();
    }

    private static String text(Message message) {
        StringBuilder text = new StringBuilder();
        for (Content part : message.content()) {
            if (part instanceof Content.Text value) text.append(value.text());
            else if (part instanceof Content.ToolResult value) text.append(value.text());
        }
        return text.toString();
    }

    private static void requireSupported(Message message) {
        for (Content part : message.content()) {
            boolean supported =
                    (message.role().equals(Role.SYSTEM) || message.role().equals(Role.USER))
                                    && part instanceof Content.Text
                            || message.role().equals(Role.ASSISTANT)
                                    && (part instanceof Content.Text
                                            || part instanceof Content.Reasoning
                                            || part instanceof Content.ToolCall)
                            || message.role().equals(Role.TOOL)
                                    && part instanceof Content.ToolResult;
            if (!supported)
                throw new UnsupportedConversation(
                        message.role().name() + " turn: " + part.getClass().getSimpleName());
        }
    }

    @Override
    public ReplyParser parser(Tokenizer tokenizer) {
        return spans.parser();
    }

    @Override
    public Optional<ReplyLanguage.Selection> constrainedReply(String contentGbnf) {
        return Optional.of(spans.constrained(contentGbnf));
    }

    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(spans.forcedCall(callableTools, Tool::name));
    }

    @Override
    public int defaultReasoningBudget(int maxTokens) {
        return -1;
    }
}
