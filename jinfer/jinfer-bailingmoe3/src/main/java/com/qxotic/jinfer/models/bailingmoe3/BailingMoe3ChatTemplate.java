package com.qxotic.jinfer.models.bailingmoe3;

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

/** Native encoder and reply codec for Ling's Bailing V3 chat format. */
public final class BailingMoe3ChatTemplate implements ChatTemplate {
    private static final String THINK_OPEN = "<think>";
    private static final String THINK_CLOSE = "</think>";
    private static final String CALL_OPEN = "<tool_call>";
    private static final String CALL_CLOSE = "</tool_call>";
    private static final String TURN_END = "<|role_end|>";
    private static final String TOOL_INSTRUCTIONS =
            "# Tools\n\n"
                    + "You may call one or more functions to assist with the user query.\n\n"
                    + "You are provided with function signatures within <tools></tools> XML tags:"
                    + "\n<tools>";
    private static final String CALL_INSTRUCTIONS =
            "\n</tools>\n\n"
                    + "If none of the functions can be used, point it out. If the given question"
                    + " lacks the parameters required by the function, also point it out.\n"
                    + "If you need to use a function, for each function call, output the function"
                    + " name and arguments within the following XML format:\n"
                    + "<tool_call>{function-name}\n"
                    + "<arg_key>{arg-key-1}</arg_key>\n"
                    + "<arg_value>{arg-value-1}</arg_value>\n"
                    + "<arg_key>{arg-key-2}</arg_key>\n"
                    + "<arg_value>{arg-value-2}</arg_value>\n"
                    + "...\n</tool_call>\n";

    private final Tokenizer tokenizer;
    private final int thinkOpen;
    private final int thinkClose;
    private final ReplyLanguage.Spans spans;

    public BailingMoe3ChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
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
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        Objects.requireNonNull(conversation, "conversation");
        List<Message> messages = conversation.messages();
        if (messages.isEmpty()) throw new UnsupportedConversation("empty conversation");
        for (Message message : messages) requireSupported(message);

        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        Message first = messages.getFirst();
        String system = first.role().equals(Role.SYSTEM) ? text(first) : "";
        writeSystem(
                out,
                system,
                !conversation.tools().isEmpty(),
                conversation.tools(),
                conversation.thinking());
        out.flush();

        for (int i = 0; i < messages.size(); i++) {
            Message message = messages.get(i);
            if (i == 0 && message.role().equals(Role.SYSTEM)) continue;
            if (message.role().equals(Role.USER)) {
                out.trusted("<role>HUMAN</role>").text(text(message)).trusted(TURN_END);
            } else if (message.role().equals(Role.SYSTEM)) {
                out.trusted("<role>SYSTEM</role>").text(text(message)).trusted(TURN_END);
            } else if (message.role().equals(Role.ASSISTANT)) {
                writeAssistant(out, message);
            } else {
                writeToolResults(out, messages, i, message);
            }
            out.flush();
        }

        out.trusted("<role>ASSISTANT</role>\n").id(thinkOpen);
        IntSequence replyPrefix;
        if (conversation.thinking()) {
            replyPrefix = IntSequence.of(thinkOpen);
        } else {
            out.id(thinkClose);
            replyPrefix = IntSequence.of(thinkOpen, thinkClose);
        }
        out.finish();

        ReplyParser parser = spans.parser();
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    private static void writeSystem(
            PromptWriter out, String system, boolean hasTools, List<Tool> tools, boolean thinking) {
        out.trusted("<role>SYSTEM</role>");
        if (hasTools) {
            if (!system.isEmpty()) out.text(system).text("\n");
            out.trusted(TOOL_INSTRUCTIONS);
            for (Tool tool : tools) {
                out.text("\n").text(ToolCallSyntax.jinjaJson(toolEnvelope(tool)));
            }
            out.trusted(CALL_INSTRUCTIONS);
            if (!declaresThinking(system))
                out.text(thinking ? "detailed thinking on" : "detailed thinking off");
        } else if (!system.isEmpty()) {
            out.text(system);
            if (!declaresThinking(system)) {
                out.text(thinking ? "\ndetailed thinking on" : "\ndetailed thinking off");
            }
        } else {
            out.text(thinking ? "detailed thinking on" : "detailed thinking off");
        }
        out.trusted(TURN_END);
    }

    private static Map<String, Object> toolEnvelope(Tool tool) {
        if (tool.definition().containsKey("function")) return tool.definition();
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("type", "function");
        envelope.put("function", tool.definition());
        return envelope;
    }

    private static boolean declaresThinking(String system) {
        return system.contains("detailed thinking on") || system.contains("detailed thinking off");
    }

    private void writeAssistant(PromptWriter out, Message message) {
        AssistantText assistant = assistantText(message);
        out.trusted("<role>ASSISTANT</role>\n").id(thinkOpen);
        if (!assistant.reasoning().isEmpty()) out.text(assistant.reasoning());
        out.id(thinkClose).text(assistant.visible());

        boolean first = true;
        for (Content part : message.content()) {
            if (!(part instanceof Content.ToolCall call)) continue;
            if (!first || !assistant.visible().isEmpty()) out.text("\n");
            writeCall(out, call);
            first = false;
        }
        out.trusted(TURN_END);
    }

    private static void writeCall(PromptWriter out, Content.ToolCall call) {
        out.trusted(CALL_OPEN).text(call.name());
        if (!call.arguments().isEmpty()) out.text("\n");
        for (Map.Entry<String, Object> argument : call.arguments().entrySet()) {
            out.trusted("<arg_key>")
                    .text(argument.getKey())
                    .trusted("</arg_key>\n<arg_value>")
                    .text(
                            argument.getValue() instanceof String text
                                    ? text
                                    : ToolCallSyntax.jinjaJson(argument.getValue()))
                    .trusted("</arg_value>");
        }
        out.trusted("\n" + CALL_CLOSE);
    }

    private static void writeToolResults(
            PromptWriter out, List<Message> messages, int index, Message message) {
        if (index == 0 || !messages.get(index - 1).role().equals(Role.TOOL))
            out.trusted("<role>OBSERVATION</role>");
        for (Content part : message.content()) {
            if (part instanceof Content.ToolResult result) {
                out.trusted("\n<tool_response>\n")
                        .text(result.text())
                        .trusted("\n</tool_response>");
            }
        }
        if (index == messages.size() - 1 || !messages.get(index + 1).role().equals(Role.TOOL))
            out.trusted(TURN_END);
    }

    private static AssistantText assistantText(Message message) {
        StringBuilder rendered = new StringBuilder();
        for (Content part : message.content()) {
            if (part instanceof Content.Text text) rendered.append(text.text());
            else if (part instanceof Content.Reasoning reasoning)
                rendered.append(THINK_OPEN).append(reasoningText(reasoning)).append(THINK_CLOSE);
        }
        String content = rendered.toString();
        int firstClose = content.indexOf(THINK_CLOSE);
        if (firstClose < 0) return new AssistantText("", content);
        String before = stripTrailingNewlines(content.substring(0, firstClose));
        int lastOpen = before.lastIndexOf(THINK_OPEN);
        String reasoning =
                stripLeadingNewlines(
                        before.substring(lastOpen < 0 ? 0 : lastOpen + THINK_OPEN.length()));
        int lastClose = content.lastIndexOf(THINK_CLOSE);
        return new AssistantText(
                stripNewlines(reasoning),
                stripLeadingNewlines(content.substring(lastClose + THINK_CLOSE.length())));
    }

    private static String stripNewlines(String text) {
        return stripTrailingNewlines(stripLeadingNewlines(text));
    }

    private static String stripLeadingNewlines(String text) {
        int at = 0;
        while (at < text.length() && text.charAt(at) == '\n') at++;
        return text.substring(at);
    }

    private static String stripTrailingNewlines(String text) {
        int at = text.length();
        while (at > 0 && text.charAt(at - 1) == '\n') at--;
        return text.substring(0, at);
    }

    private static String text(Message message) {
        StringBuilder text = new StringBuilder();
        for (Content part : message.content()) {
            if (part instanceof Content.Text value) text.append(value.text());
        }
        return text.toString();
    }

    private static String reasoningText(Content.Reasoning reasoning) {
        StringBuilder text = new StringBuilder();
        for (Content part : reasoning.content()) {
            if (!(part instanceof Content.Text value))
                throw new UnsupportedConversation(
                        "reasoning contains " + part.getClass().getSimpleName());
            text.append(value.text());
        }
        return text.toString();
    }

    private static void requireSupported(Message message) {
        boolean standardRole =
                message.role().equals(Role.SYSTEM)
                        || message.role().equals(Role.USER)
                        || message.role().equals(Role.ASSISTANT)
                        || message.role().equals(Role.TOOL);
        if (!standardRole) throw new UnsupportedConversation("role " + message.role().name());
        if (message.role().equals(Role.TOOL) && message.content().isEmpty())
            throw new UnsupportedConversation("empty tool turn");

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
    public Optional<ReplyLanguage.Selection> constrainedReply(
            String contentGbnf, List<Tool> callableTools) {
        return Optional.of(spans.constrainedAuto(contentGbnf, !callableTools.isEmpty()));
    }

    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(spans.forcedCall(callableTools, Tool::name));
    }

    private record AssistantText(String reasoning, String visible) {}
}
