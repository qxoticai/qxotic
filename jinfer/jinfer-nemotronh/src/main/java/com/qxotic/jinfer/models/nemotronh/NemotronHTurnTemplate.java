package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * Hand-written Nemotron-H chat framing (ChatML dialect), matching the GGUF chat_template's shape
 * ({@code truncate_history_thinking=true} default), tools included: declarations render into the
 * system turn and calls/results frame natively (see {@link #encode}), replies parse the {@code
 * <tool_call>} span grammar (see {@link #parser}).
 *
 * <p>Layout: no bos; per turn {@code <|im_start|>{role}\n{content}<|im_end|>\n}. The template
 * ALWAYS renders a system turn - when the conversation lacks one it injects a default persona - so
 * {@link #encode} prepends {@link #DEFAULT_SYSTEM} when the first message is not a system turn
 * (incremental drivers supply their own system turn; both existing harnesses do).
 *
 * <p>Historical assistant turns match the template's truncation: content with neither think marker
 * is prefixed with an empty {@code <think></think>}; content with both keeps only the text after
 * the LAST {@code </think>} behind the empty pair; the result is trimmed. The think pair is emitted
 * as trusted special ids; everything else is ONE contiguous plain {@link Tokenizer#encode} run per
 * span between specials (that is how a rendered template tokenizes), so conversation text cannot
 * mint control tokens. Unclosed-{@code <think>} content (the template's "broken thought" path) is
 * passed through plain-encoded - a documented divergence from the render+rescan oracle.
 *
 * <p>Generation prompt: {@code <|im_start|>assistant\n<think>\n} (thinking) or {@code
 * <|im_start|>assistant\n<think></think>} - no trailing newline - matching the template's {@code
 * enable_thinking} branches.
 */
public final class NemotronHTurnTemplate implements TurnTemplate {

    public static final String DEFAULT_SYSTEM =
            "You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.";

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
    private final IntSequence newline; // encode("\n"), constant
    private final IntSequence assistantNl; // encode("assistant\n"), constant
    private final List<Batch> genThinking, genDirect; // generation prompts, encoded once
    private final List<Batch> closeTurn; // <|im_end|>\n, constant

    public NemotronHTurnTemplate(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        this.imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        this.think = SpecialTokens.require(tokenizer, "<think>");
        this.endThink = SpecialTokens.require(tokenizer, "</think>");
        this.toolCall = SpecialTokens.find(tokenizer, "<tool_call>").orElse(-1);
        this.endToolCall = SpecialTokens.find(tokenizer, "</tool_call>").orElse(-1);
        this.toolResponse = SpecialTokens.find(tokenizer, "<tool_response>").orElse(-1);
        this.endToolResponse = SpecialTokens.find(tokenizer, "</tool_response>").orElse(-1);
        this.newline = tokenizer.encode("\n");
        this.assistantNl = tokenizer.encode("assistant\n");
        IntSequence head =
                IntSequence.of(imStart).concat(assistantNl).concat(IntSequence.of(think));
        IntSequence thinking = head.concat(newline); // <|im_start|>assistant\n<think>\n
        this.genThinking = List.of(Batch.prefill(thinking.toArray()));
        // <|im_start|>assistant\n<think></think>  (no newline)
        IntSequence direct = head.concat(IntSequence.of(endThink));
        this.genDirect = List.of(Batch.prefill(direct.toArray()));
        IntSequence close = IntSequence.of(imEnd).concat(newline);
        this.closeTurn = List.of(Batch.prefill(close.toArray()));
    }

    /**
     * No unconditional tokens (no bos). The default-system injection lives in {@link #normalize}.
     */
    @Override
    public List<Batch> conversationStart() {
        return List.of();
    }

    /**
     * The template unconditionally renders a system turn: inject the default when absent, so every
     * caller (whole-render AND turn-by-turn drivers) frames identically.
     */
    @Override
    public List<Message> normalize(List<Message> conversation) {
        if (!conversation.isEmpty() && conversation.get(0).role().equals(Role.SYSTEM)) {
            return conversation;
        }
        List<Message> out = new ArrayList<>(conversation.size() + 1);
        out.add(Message.system(DEFAULT_SYSTEM));
        out.addAll(conversation);
        return out;
    }

    @Override
    public List<Batch> encodeTurn(Message m) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(imStart);
        if (m.role().equals(Role.ASSISTANT)) {
            String c = m.textOnly();
            int lastClose = c.lastIndexOf("</think>");
            ids.addAll(assistantNl);
            if (c.contains("<think>") == (lastClose >= 0)) {
                // framed (both markers -> keep the tail after the last </think>) or plain (no
                // markers -> the whole content): emit the empty pair, then the text with its
                // trailing whitespace stripped (leading whitespace survives, as in the template)
                String rest =
                        (lastClose >= 0 ? c.substring(lastClose + "</think>".length()) : c)
                                .stripTrailing();
                ids.add(think);
                ids.add(endThink);
                if (!rest.isEmpty()) ids.addAll(tokenizer.encode(rest));
            } else {
                // "broken thought" (unpaired marker): plain-encoded passthrough, fully stripped
                String rest = c.strip();
                if (!rest.isEmpty()) ids.addAll(tokenizer.encode(rest));
            }
        } else {
            // user/system: role header + content is ONE contiguous run between the specials
            ids.addAll(tokenizer.encode(m.role().name() + "\n" + m.textOnly()));
        }
        ids.add(imEnd);
        ids.addAll(newline);
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return thinking ? genThinking : genDirect;
    }

    /** Closes the assistant turn: {@code <|im_end|>\n} (the stop token is never ingested). */
    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    // The format-instructions constant, split where its literal <tool_call> spellings sit: those
    // are template-authored and emit as trusted ids (matching the render+rescan). The reminder
    // line's own <tool_call></tool_call> mention splits INSTRUCTIONS_TAIL the same way.
    static final String INSTRUCTIONS_HEAD =
            "\n\nIf you choose to call a function ONLY reply in the following format with NO"
                    + " suffix:\n\n";
    static final String INSTRUCTIONS_EXAMPLE =
            "\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
                    + "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the"
                    + " second parameter\nthat can span\nmultiple lines\n</parameter>\n"
                    + "</function>\n";
    static final String INSTRUCTIONS_REMINDER =
            "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an"
                    + " inner <function=...></function> block must be nested within ";
    static final String INSTRUCTIONS_TAIL =
            " XML tags\n"
                + "- Required parameters MUST be specified\n"
                + "- You may provide optional reasoning for your function call in natural language"
                + " BEFORE the function call, but NOT after\n"
                + "- If there is no function call available, answer the question like normal with"
                + " your current knowledge and do not tell the user about function calls\n"
                + "</IMPORTANT>";

    /**
     * The codec face, tools included - the template's whole-conversation flow. Plain conversations
     * keep the oracle-validated per-turn fold; tool-bearing ones render natively: declarations +
     * format instructions inside the system turn (the instruction text's literal {@code
     * <tool_call>} spellings become trusted ids, matching the render+rescan), assistant call turns
     * as {@code <think>...</think>} content then one {@code <tool_call>} block per call, tool
     * results folded into a single {@code user} turn of {@code <tool_response>} blocks, and the
     * template's history-thinking truncation for call turns before the last user message.
     */
    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        if (conversation.tools().isEmpty() && plainShape(msgs)) {
            return TurnTemplate.super.encode(conversation);
        }
        TurnTemplate.requireToolShapes(msgs);
        if (toolCall < 0) {
            throw new UnsupportedConversation("tools need the <tool_call> markers in the vocab");
        }

        Message sys =
                !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM) ? msgs.get(0) : null;
        String sysText = sys != null ? sys.textOnly() : DEFAULT_SYSTEM;
        List<Message> loop = sys != null ? msgs.subList(1, msgs.size()) : msgs;
        int lastUser = -1;
        for (int i = 0; i < loop.size(); i++) {
            if (loop.get(i).role().equals(Role.USER)) lastUser = i;
        }

        IntSequence.Builder ids = IntSequence.newBuilder();
        StringBuilder text = new StringBuilder();
        // system turn: message text, then declarations + instructions when tools are offered
        ids.add(imStart);
        text.append("system\n").append(sysText);
        if (!conversation.tools().isEmpty()) {
            if (!sysText.isEmpty()) text.append("\n\n");
            text.append(NemotronToolSyntax.declarations(conversation.tools()));
            text.append(INSTRUCTIONS_HEAD);
            emit(ids, text, toolCall);
            text.append(INSTRUCTIONS_EXAMPLE);
            emit(ids, text, endToolCall);
            text.append(INSTRUCTIONS_REMINDER);
            emit(ids, text, toolCall);
            emit(ids, text, endToolCall);
            text.append(INSTRUCTIONS_TAIL);
        }
        emit(ids, text, imEnd);
        text.append('\n');

        for (int i = 0; i < loop.size(); i++) {
            Message m = loop.get(i);
            if (m.role().equals(Role.TOOL)) {
                if (i == 0 || !loop.get(i - 1).role().equals(Role.TOOL)) {
                    emit(ids, text, imStart);
                    text.append("user\n");
                }
                emit(ids, text, toolResponse);
                text.append('\n');
                for (Part p : m.content()) {
                    if (p instanceof Part.ToolResult r) text.append(r.text());
                }
                text.append('\n');
                emit(ids, text, endToolResponse);
                text.append('\n');
                boolean nextIsTool =
                        i + 1 < loop.size() && loop.get(i + 1).role().equals(Role.TOOL);
                if (!nextIsTool) {
                    emit(ids, text, imEnd);
                    text.append('\n');
                }
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                emit(ids, text, imStart);
                text.append(m.role().name()).append('\n').append(m.textOnly());
                emit(ids, text, imEnd);
                text.append('\n');
                continue;
            }
            List<Part.ToolCall> calls =
                    m.content().stream()
                            .filter(p -> p instanceof Part.ToolCall)
                            .map(p -> (Part.ToolCall) p)
                            .toList();
            emit(ids, text, imStart);
            text.append("assistant\n");
            Part.Reasoning thinking = m.reasoning();
            boolean truncate = i < lastUser; // truncate_history_thinking, template default true
            if (thinking != null && !truncate) {
                emit(ids, text, think);
                text.append('\n').append(thinking.text()).append('\n');
                emit(ids, text, endThink);
                // the "\n" after </think> survives the template's trim only when text follows
                // (the call branch's own '\n' re-adds it before <tool_call> blocks)
                String tail = m.text().stripTrailing();
                if (!tail.isEmpty()) text.append('\n').append(tail);
            } else {
                // truncated (or no reasoning): the empty pair then the text - trailing-trimmed
                // when kept as-is, fully stripped through the template's truncation split
                emit(ids, text, think);
                emit(ids, text, endThink);
                String tail = truncate ? m.text().strip() : m.text().stripTrailing();
                if (!tail.isEmpty()) text.append(tail);
            }
            if (!calls.isEmpty()) {
                text.append('\n'); // the call branch appends '\n' after the content
                for (Part.ToolCall call : calls) {
                    emit(ids, text, toolCall);
                    text.append(NemotronToolSyntax.call(call));
                    emit(ids, text, endToolCall);
                    text.append('\n');
                }
            }
            emit(ids, text, imEnd);
            text.append('\n');
        }
        flush(ids, text);
        List<Batch> out = new ArrayList<>();
        out.add(Batch.prefill(ids.build().toArray()));
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }

    /** Flushes the pending text run plainly, then emits one trusted id. */
    private void emit(IntSequence.Builder ids, StringBuilder text, int id) {
        flush(ids, text);
        ids.add(id);
    }

    private void flush(IntSequence.Builder ids, StringBuilder text) {
        if (text.isEmpty()) return;
        ids.addAll(tokenizer.encode(text.toString()));
        text.setLength(0);
    }

    private static boolean plainShape(List<Message> msgs) {
        for (Message m : msgs) {
            for (Part p : m.content()) {
                if (!(p instanceof Part.Text)) return false;
            }
        }
        return true;
    }

    /**
     * Tool calls parse natively: the reply's {@code <tool_call>}/{@code </tool_call>} trusted ids
     * claim XML-function payloads ({@code <function=NAME><parameter=K>...} - the grammar Nemotron
     * shares with Qwen 3.5).
     */
    @Override
    public ReplyParser parser() {
        if (toolCall < 0) {
            return ReplyParser.spans(tokenizer); // older vocabs without the tool markers
        }
        return ReplyParser.spans(
                tokenizer, "<tool_call>", "</tool_call>", ToolCallSyntax::parseFunctionXml);
    }

    /** Forced calls seed {@code <tool_call>} (seeding only - no pin hook yet). */
    @Override
    public int[] callSeed() {
        return toolCall < 0 ? new int[0] : new int[] {toolCall};
    }

    /** The generation prompt opens the think span (or its closed pair): pre-feed it. */
    @Override
    public int[] replySeed(boolean thinking) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(think);
        if (thinking) ids.addAll(newline);
        else ids.add(endThink);
        return ids.build().toArray();
    }
}
