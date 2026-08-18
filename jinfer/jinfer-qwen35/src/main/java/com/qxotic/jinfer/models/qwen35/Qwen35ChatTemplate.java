package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jinfer.boundary.MediaProjector;
import com.qxotic.jinfer.boundary.Multimodal;
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
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Qwen 3.5 chat framing (ChatML dialect), token-exact with the GGUF's Jinja chat_template: NO bos,
 * per turn {@code <|im_start|>{role}\n{content|trim}<|im_end|>\n}, generation prompt {@code
 * <|im_start|>assistant\n} plus the thinking scaffold ({@code <think>\n} to reason, the pre-closed
 * {@code <think>\n\n</think>\n\n} to answer directly).
 *
 * <p>Two shapes: plain conversations fold per turn (turn-stable blocks, the cached-prompt law);
 * tool-bearing ones render the template's whole flow - declarations ({@code # Tools ... <tools>}
 * with each tool's {@code tojson}) plus the format instructions inside the system turn, assistant
 * call turns as content then one {@code <tool_call>} XML function block per call, tool results
 * folded into a single {@code user} turn of {@code <tool_response>} blocks, and reasoning kept only
 * on assistant turns after the last real user query.
 */
public final class Qwen35ChatTemplate implements ChatTemplate {

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

    private final Tokenizer tokenizer;
    private final Multimodal media;
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    private final int visionOpen; // <|vision_start|>
    private final int visionClose; // <|vision_end|>
    private final IntSequence seedThinking; // <think>\n
    private final IntSequence seedDirect; // <think>\n\n</think>\n\n

    public Qwen35ChatTemplate(Qwen35 model) {
        this(model.tokenizer(), model);
    }

    public Qwen35ChatTemplate(Tokenizer tokenizer) {
        this(tokenizer, null);
    }

    public Qwen35ChatTemplate(Tokenizer tokenizer, Multimodal media) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.media = media;
        imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        think = SpecialTokens.require(tokenizer, "<think>");
        endThink = SpecialTokens.require(tokenizer, "</think>");
        visionOpen = SpecialTokens.require(tokenizer, "<|vision_start|>");
        visionClose = SpecialTokens.require(tokenizer, "<|vision_end|>");
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
        List<Message> msgs = conversation.messages();
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        if (conversation.tools().isEmpty() && plainShape(msgs)) {
            for (Message m : msgs) {
                writePlainTurn(out, m);
                out.flush();
            }
        } else {
            requireToolShapes(msgs);
            SpecialTokens.require(tokenizer, "<tool_call>");
            SpecialTokens.require(tokenizer, "</tool_call>");
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
     * {@code <|im_start|>{role}\n{content|trim}<|im_end|>\n} - one contiguous run per turn.
     * Media-bearing turns write parts verbatim with images framed as the template's {@code
     * <|vision_start|> rows <|vision_end|>}.
     */
    private void writePlainTurn(PromptWriter out, Message m) {
        boolean hasMedia = m.content().stream().anyMatch(Content.Media.class::isInstance);
        if (hasMedia) {
            out.id(imStart).text(m.role().name() + "\n");
            for (Content part : m.content()) {
                switch (part) {
                    case Content.Text text -> out.text(text.text());
                    case Content.Media value -> writeMedia(out, value);
                    default ->
                            throw new UnsupportedConversation(
                                    "unsupported content type in Qwen3.5 message");
                }
            }
            out.id(imEnd).text("\n");
            return;
        }
        String content = m.text().strip();
        if (m.role().equals(Role.ASSISTANT)) content = stripThinking(content);
        out.id(imStart).text(m.role().name() + "\n" + content).id(imEnd).text("\n");
    }

    private void writeMedia(PromptWriter out, Content.Media content) {
        if (!(content.value() instanceof Media.Image image))
            throw new UnsupportedConversation("Qwen3.5 vision supports images only");
        MediaProjector<Media.Image> projector =
                media == null ? null : media.projector(Media.Image.class).orElse(null);
        if (projector == null)
            throw new UnsupportedConversation(
                    "image input is not supported by this model (attach --with"
                            + " media=<mmproj.gguf>)");
        out.cachedMedia(
                image,
                content.contentKey(),
                encoded ->
                        encoded.id(visionOpen)
                                .media(image, content.contentKey(), projector, false)
                                .id(visionClose));
    }

    /** Best-effort media positions via the modality's embedder plan (no encoding). */
    @Override
    public int mediaPositions(Media m) {
        MediaProjector<Media.Image> projector =
                media == null ? null : media.projector(Media.Image.class).orElse(null);
        if (projector == null)
            throw new UnsupportedOperationException("image input is not supported by this model");
        if (m instanceof Media.Image img) return projector.positions(img);
        throw new UnsupportedOperationException("Qwen3.5 vision supports images only");
    }

    /** {@code <|im_start|>assistant\n} + the thinking scaffold. */
    private void writeGenerationPrompt(PromptWriter out, boolean thinking) {
        out.id(imStart).text("assistant\n").id(think);
        if (thinking) {
            out.text("\n");
        } else {
            out.text("\n\n").id(endThink).text("\n\n");
        }
    }

    /** The template's whole tool flow: declarations, call turns, folded tool responses. */
    private void writeToolConversation(
            PromptWriter out, Conversation conversation, List<Message> msgs) {
        Message sys = leadingSystem(msgs);
        if (!conversation.tools().isEmpty()) {
            out.id(imStart).text("system\n");
            out.text("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for (Tool tool : conversation.tools()) {
                out.text("\n").text(ToolCallSyntax.jinjaJson(tool.definition()));
            }
            out.text("\n</tools>");
            out.trusted(TOOL_INSTRUCTIONS);
            if (sys != null) {
                String sysText = sys.text().strip();
                if (!sysText.isEmpty()) out.text("\n\n").text(sysText);
            }
            out.id(imEnd).text("\n");
        } else if (sys != null) {
            out.id(imStart).text("system\n").text(sys.text().strip()).id(imEnd).text("\n");
        }

        // the template's walk-back: the last USER turn that is not a <tool_response> wrapper;
        // assistant turns AFTER it keep their reasoning, all earlier ones drop it entirely
        int lastQuery = -1;
        for (int i = msgs.size() - 1; i >= 0; i--) {
            Message m = msgs.get(i);
            if (!m.role().equals(Role.USER)) continue;
            String c = m.text().strip();
            if (c.startsWith("<tool_response>") && c.endsWith("</tool_response>")) continue;
            lastQuery = i;
            break;
        }

        for (int i = 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.SYSTEM)) continue; // rendered above (template: first only)
            if (m.role().equals(Role.TOOL)) {
                if (i == 0 || !msgs.get(i - 1).role().equals(Role.TOOL)) {
                    out.id(imStart).text("user");
                }
                out.trusted("\n<tool_response>\n");
                StringBuilder result = new StringBuilder();
                for (Content part : m.content()) {
                    switch (part) {
                        case Content.ToolResult r -> result.append(r.text());
                        case Content.Text t -> result.append(t.text());
                        default -> {}
                    }
                }
                out.text(result.toString().strip());
                out.trusted("\n</tool_response>");
                boolean nextIsTool =
                        i + 1 < msgs.size() && msgs.get(i + 1).role().equals(Role.TOOL);
                if (!nextIsTool) out.id(imEnd).text("\n");
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                out.id(imStart).text(m.role().name()).text("\n").text(m.text().strip());
                out.id(imEnd).text("\n");
                continue;
            }
            String content = m.text().strip();
            String reasoningText = reasoningText(m);
            if (reasoningText == null) {
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
            out.id(imStart).text("assistant\n");
            if (i > lastQuery) {
                out.id(think)
                        .text("\n")
                        .text(reasoningText)
                        .text("\n")
                        .id(endThink)
                        .text("\n\n")
                        .text(content);
            } else {
                out.text(content);
            }
            List<Content.ToolCall> calls = callsOf(m);
            for (int c = 0; c < calls.size(); c++) {
                Content.ToolCall call = calls.get(c);
                out.text(c == 0 ? (content.isEmpty() ? "" : "\n\n") : "\n");
                out.trusted("<tool_call>\n");
                out.text("<function=" + call.name() + ">\n");
                for (Map.Entry<String, Object> arg : call.arguments().entrySet()) {
                    out.text("<parameter=" + arg.getKey() + ">\n");
                    out.text(ToolCallSyntax.jinjaValue(arg.getValue()));
                    out.text("\n</parameter>\n");
                }
                out.text("</function>\n");
                out.trusted("</tool_call>");
            }
            out.id(imEnd).text("\n");
        }
    }

    private static Message leadingSystem(List<Message> msgs) {
        return !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM) ? msgs.get(0) : null;
    }

    private static String reasoningText(Message m) {
        for (Content part : m.content()) {
            if (part instanceof Content.Reasoning reasoning) return reasoning.text().strip();
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
                if (!(part instanceof Content.Text) && !(part instanceof Content.Media))
                    return false;
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

    /**
     * The template keeps only the text after the last {@code </think>}, leading newlines stripped:
     * {@code content.split('</think>')[-1].lstrip('\n')}.
     */
    private static String stripThinking(String content) {
        int at = content.lastIndexOf("</think>");
        if (at < 0) return content;
        return stripLeadingNl(content.substring(at + "</think>".length()));
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
