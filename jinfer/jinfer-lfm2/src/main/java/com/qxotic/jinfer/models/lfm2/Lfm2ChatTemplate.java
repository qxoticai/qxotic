package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.List;

/**
 * Hand-written LFM2.5 chat codec (ChatML dialect), byte-exact with the GGUF's Jinja {@code
 * tokenizer.chat_template} and validated against it offline - the native reference implementation
 * of the {@link ChatTemplate} codec (the per-turn {@link TurnTemplate} view remains implemented as
 * the building blocks {@code encode} composes, and for the turn-aligned test scenarios).
 *
 * <p>Layout: {@code <|startoftext|>} once, then per turn {@code
 * <|im_start|>{role}\n{content}<|im_end|>\n}, generation prompt {@code <|im_start|>assistant\n}.
 * The role header and the content are tokenized as ONE contiguous plain-encoded run - that is how
 * the rendered template tokenizes (specials force the only splits), and BPE merges across the
 * header/content boundary, so encoding them separately would drift.
 *
 * <p>Two domains: {@code <|startoftext|>}/{@code <|im_start|>}/{@code <|im_end|>} are emitted as
 * trusted ids; everything else goes through plain {@link Tokenizer#encode} so conversation text can
 * never mint control tokens. Matching the template, a re-rendered assistant turn keeps only the
 * text after its last {@code </think>} (trimmed).
 *
 * <p>VERBATIM SPLICE: an assistant message whose model-produced parts all carry verbatim token ids
 * (a reply built by this codec's own {@link ReplyParser}) replays them exactly - {@code
 * generationPrompt ++ payload ids (delimiters re-emitted as trusted ids) ++ closeTurn} - so
 * re-encoding an extended conversation reproduces the generated KV token-for-token (the round-trip
 * law; KV continuity for the in-process agentic/CLI loop). Wire-echoed history never carries
 * verbatim ids and always re-renders faithfully per the reference template.
 *
 * <p>Tools weld into the system turn ({@code List of tools: [...]}); assistant tool-call turns
 * append {@code <|tool_call_start|>[pythonic]<|tool_call_end|>}; tool results are a {@code tool}
 * turn. Out of scope, matching the model: media parts (text-only port).
 */
public final class Lfm2ChatTemplate implements TurnTemplate {

    private final Tokenizer tokenizer;
    private final int bos; // <|startoftext|>
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int tcStart; // <|tool_call_start|>
    private final int tcEnd; // <|tool_call_end|>
    private final Integer thinkOpen; // <think>, null when the vocab lacks it
    private final Integer thinkClose; // </think>
    private final int[] newline; // encode("\n"), constant
    private final IntSequence generationPromptIds; // <|im_start|>assistant\n, constant
    private final List<Batch> generationPrompt;
    private final IntSequence closeTurnIds; // <|im_end|>\n, constant
    private final List<Batch> closeTurn;

    public Lfm2ChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.bos = SpecialTokens.require(tokenizer, "<|startoftext|>");
        this.imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        this.imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        this.tcStart = SpecialTokens.require(tokenizer, "<|tool_call_start|>");
        this.tcEnd = SpecialTokens.require(tokenizer, "<|tool_call_end|>");
        java.util.OptionalInt open = SpecialTokens.find(tokenizer, "<think>");
        java.util.OptionalInt close = SpecialTokens.find(tokenizer, "</think>");
        this.thinkOpen = open.isPresent() ? open.getAsInt() : null;
        this.thinkClose = close.isPresent() ? close.getAsInt() : null;
        this.newline = tokenizer.encode("\n").toArray();
        this.generationPromptIds = IntSequence.of(imStart).concat(tokenizer.encode("assistant\n"));
        this.generationPrompt = List.of(Batch.prefill(generationPromptIds.toArray()));
        this.closeTurnIds = IntSequence.of(imEnd).concat(tokenizer.encode("\n"));
        this.closeTurn = List.of(Batch.prefill(closeTurnIds.toArray()));
    }

    // ---- ChatTemplate: the codec ----

    @Override
    public List<Batch> encode(Conversation conversation) {
        requireSupported(conversation);
        List<Batch> out = new ArrayList<>();
        List<Message> turns = conversation.messages();
        if (!conversation.tools().isEmpty()) {
            turns = TurnTemplate.encodePreamble(this, turns, conversation.tools(), out);
        } else {
            out.addAll(conversationStart());
        }
        for (Message message : turns) {
            out.addAll(spliceable(message) ? spliceTurn(message) : encodeTurn(message));
        }
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }

    @Override
    public ReplyParser parser() {
        return ReplyParser.spans(
                tokenizer, "<|tool_call_start|>", "<|tool_call_end|>", ToolCallSyntax::parseBlock);
    }

    /** The part shapes this port frames byte-exactly; anything else punts to the whole render. */
    private static void requireSupported(Conversation conversation) {
        for (Message message : conversation.messages()) {
            boolean toolTurn = message.role().equals(Role.TOOL);
            boolean assistant = message.role().equals(Role.ASSISTANT);
            for (Part part : message.content()) {
                boolean ok =
                        switch (part) {
                            case Part.Text t -> true;
                            case Part.ToolResult r -> toolTurn;
                            case Part.ToolCall c -> assistant;
                            case Part.Reasoning r -> assistant;
                            case Part.Blob b -> false; // text-only model
                        };
                if (!ok)
                    throw new UnsupportedConversation(
                            message.role().name() + " turn: " + part.getClass().getSimpleName());
            }
        }
    }

    // ---- verbatim splice (the round-trip law) ----

    /**
     * An assistant turn replays verbatim only when EVERY model-produced part carries its generated
     * ids (a decoder-built reply) - a single caller-authored or stripped part re-renders the whole
     * turn faithfully instead.
     */
    private boolean spliceable(Message message) {
        if (!message.role().equals(Role.ASSISTANT) || message.content().isEmpty()) return false;
        return partsCarryVerbatim(message.content());
    }

    private boolean partsCarryVerbatim(List<Part> parts) {
        for (Part part : parts) {
            boolean ok =
                    switch (part) {
                        case Part.Text t -> t.verbatim() != null;
                        case Part.ToolCall c -> c.verbatim() != null;
                        case Part.Reasoning r ->
                                thinkOpen != null
                                        && thinkClose != null
                                        && r.verbatim() != null
                                        && partsCarryVerbatim(r.content());
                        default -> false;
                    };
            if (!ok) return false;
        }
        return true;
    }

    /**
     * {@code generationPrompt ++ payload ids ++ closeTurn}: exactly the token stream generation
     * produced, with the structural delimiters the decoder consumed re-emitted as trusted ids.
     */
    private List<Batch> spliceTurn(Message message) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.addAll(generationPromptIds);
        spliceParts(message.content(), ids);
        ids.addAll(closeTurnIds);
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    private void spliceParts(List<Part> parts, IntSequence.Builder ids) {
        for (Part part : parts) {
            switch (part) {
                case Part.Text t -> ids.addAll(t.verbatim());
                case Part.ToolCall c -> {
                    ids.add(tcStart);
                    ids.addAll(c.verbatim());
                    ids.add(tcEnd);
                }
                case Part.Reasoning r -> {
                    ids.add(thinkOpen);
                    spliceParts(r.content(), ids);
                    ids.add(thinkClose);
                }
                default -> throw new IllegalStateException("unspliceable part " + part);
            }
        }
    }

    // ---- TurnTemplate: the per-turn building blocks ----

    @Override
    public List<Batch> conversationStart() {
        return List.of(Batch.prefill(new int[] {bos}));
    }

    /**
     * The template welds tools into the system turn: {@code system_prompt = {system content}\nList
     * of tools: [{tool0 json}, {tool1 json}]}, rendered once as a system turn. When tools are
     * present but there is no system message, a system turn is still emitted with only the tool
     * list. Each tool is its raw request JSON verbatim (Jinja {@code tool | tojson}).
     */
    @Override
    public List<Batch> conversationStart(TurnTemplate.Preamble preamble) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(bos);
        String system = preamble.system().map(Message::text).orElse("");
        if (!preamble.tools().isEmpty()) {
            StringBuilder tools = new StringBuilder("List of tools: [");
            List<Tool> list = preamble.tools();
            for (int t = 0; t < list.size(); t++) {
                if (t > 0) tools.append(", ");
                tools.append(list.get(t).rawJson());
            }
            tools.append(']');
            system = system.isEmpty() ? tools.toString() : system + "\n" + tools;
        }
        if (!system.isEmpty()) {
            ids.add(imStart);
            ids.addAll(tokenizer.encode("system\n" + system));
            ids.add(imEnd);
            for (int nl : newline) ids.add(nl);
        }
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    @Override
    public List<Batch> encodeTurn(Message message) {
        Role role = message.role();
        String content = textContent(message);
        if (role.equals(Role.ASSISTANT)) {
            content = stripThinking(content);
            // A STRUCTURED reply (Reasoning parts, not literal markers) re-renders as the
            // template would render its stripped text: everything after the think span, trimmed.
            if (message.content().stream().anyMatch(p -> p instanceof Part.Reasoning)) {
                content = content.strip();
            }
        }
        List<Part.ToolCall> calls = toolCalls(message);
        // <|im_start|>{role}\n{content}[<|tool_call_start|>[calls]<|tool_call_end|>]<|im_end|>\n
        // Each plain span (header+content, and the bracketed call list) is a separate contiguous
        // run: the trusted markers split them, exactly as encodeWithSpecialTokens rescans the
        // rendered string.
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(imStart);
        ids.addAll(tokenizer.encode(role.name() + "\n" + content));
        if (!calls.isEmpty()) {
            ids.add(tcStart);
            ids.addAll(tokenizer.encode("[" + ToolCallSyntax.renderPythonic(calls) + "]"));
            ids.add(tcEnd);
        }
        ids.add(imEnd);
        for (int nl : newline) ids.add(nl);
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    /**
     * The message's text (and, for tool turns, tool-result) parts concatenated; media and tool-call
     * parts excluded. Reasoning is dropped here - the re-render path strips thinking.
     */
    private static String textContent(Message message) {
        StringBuilder sb = new StringBuilder();
        for (Part part : message.content()) {
            if (part instanceof Part.Text t) sb.append(t.text());
            else if (part instanceof Part.ToolResult r) sb.append(r.text());
        }
        return sb.toString();
    }

    /** The tool-call parts of a (usually assistant) message, in order. */
    private static List<Part.ToolCall> toolCalls(Message message) {
        List<Part.ToolCall> calls = new ArrayList<>();
        for (Part part : message.content()) if (part instanceof Part.ToolCall c) calls.add(c);
        return calls;
    }

    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        return generationPrompt;
    }

    @Override
    public List<Batch> closeTurn() {
        return closeTurn;
    }

    /**
     * The template keeps only the text after the last {@code </think>} in re-rendered assistant
     * turns: {@code content.split("</think>")[-1] | trim}.
     */
    private static String stripThinking(String content) {
        int at = content.lastIndexOf("</think>");
        return at < 0 ? content : content.substring(at + "</think>".length()).strip();
    }
}
