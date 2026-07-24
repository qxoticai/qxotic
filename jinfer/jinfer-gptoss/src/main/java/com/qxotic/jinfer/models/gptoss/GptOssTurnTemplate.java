package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Hand-written gpt-oss (Harmony) chat framing, matching the GGUF chat_template's plain-conversation
 * shape.
 *
 * <p>Layout: a fixed system preamble once ({@code <|start|>system<|message|>{identity, cutoff,
 * date, reasoning, channels}<|end|>}), the conversation's system message as a developer block
 * ({@code <|start|>developer<|message|># Instructions\n\n{content}<|end|>}), then per turn {@code
 * <|start|>user<|message|>{content}<|end|>} and, for assistant history, {@code
 * <|start|>assistant<|channel|>final<|message|>{content}<|end|>} (the template drops CoT from
 * history - only the final channel is re-rendered). Generation prompt is {@code
 * <|start|>assistant}; the model then emits its own channel tokens ({@code
 * <|channel|>analysis<|message|>...}), so {@code thinking} is a no-op here - Harmony always
 * reasons, and the effort knob lives in the system preamble ({@code Reasoning: medium}).
 *
 * <p>Each text run between specials is ONE contiguous plain {@link Tokenizer#encode} (that is how a
 * rendered template tokenizes; specials force the only splits), and conversation content never goes
 * through special-aware encoding, so text cannot mint control tokens.
 *
 * <p>The preamble embeds the current date ({@code strftime_now("%Y-%m-%d")} in the template); it is
 * a constructor argument so tests are deterministic - the convenience constructor pins today,
 * matching what the template renders.
 */
public final class GptOssTurnTemplate implements TurnTemplate {

    static final String DEFAULT_IDENTITY =
            "You are ChatGPT, a large language model trained by OpenAI.";
    static final String DEFAULT_EFFORT = "medium";

    /** Appended to the system preamble when tools are offered (the template's tools branch). */
    static final String TOOLS_LINE =
            "\nCalls to these tools must go to the commentary channel: 'functions'.";

    private final Tokenizer tokenizer;
    private final List<Batch> conversationStart; // fixed preamble, encoded once
    private final Batch toolsPreamble; // the preamble + routing line, encoded once
    private final int[] callEpilogue; // " <|constrain|>json<|message|>", resolved once
    private final int start; // <|start|>
    private final int message; // <|message|>
    private final int channel; // <|channel|>
    private final int end; // <|end|>
    private final int call; // <|call|>

    public GptOssTurnTemplate(Tokenizer tokenizer) {
        this(tokenizer, LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")));
    }

    public GptOssTurnTemplate(Tokenizer tokenizer, String currentDate) {
        this.tokenizer = tokenizer;
        this.start = SpecialTokens.require(tokenizer, "<|start|>");
        this.message = SpecialTokens.require(tokenizer, "<|message|>");
        this.channel = SpecialTokens.require(tokenizer, "<|channel|>");
        this.end = SpecialTokens.require(tokenizer, "<|end|>");
        this.call = SpecialTokens.require(tokenizer, "<|call|>");
        String systemText =
                DEFAULT_IDENTITY
                        + "\n"
                        + "Knowledge cutoff: 2024-06\n"
                        + "Current date: "
                        + currentDate
                        + "\n\n"
                        + "Reasoning: "
                        + DEFAULT_EFFORT
                        + "\n\n"
                        + "# Valid channels: analysis, commentary, final. Channel must be included"
                        + " for every message.";
        this.conversationStart = List.of(block("system", systemText));
        this.toolsPreamble = block("system", systemText + TOOLS_LINE);
        IntSequence.Builder epilogue = IntSequence.newBuilder();
        epilogue.addAll(tokenizer.encode(" "));
        epilogue.add(SpecialTokens.require(tokenizer, "<|constrain|>"));
        epilogue.addAll(tokenizer.encode("json"));
        epilogue.add(message);
        this.callEpilogue = epilogue.build().toArray();
    }

    /** {@code <|start|>{role}<|message|>{body}<|end|>} - one channel-less block. */
    private Batch block(String role, String body) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        ids.addAll(tokenizer.encode(role));
        ids.add(message);
        ids.addAll(tokenizer.encode(body));
        ids.add(end);
        return Batch.prefill(ids.build().toArray());
    }

    /**
     * {@code <|start|>{header}<|channel|>{channelName}<|message|>{body}{close}} - one channeled
     * turn. The body splices {@code verbatim} generated ids when present (model-exact bytes),
     * otherwise re-encodes {@code body} plainly.
     */
    private Batch turn(
            String header, String channelName, String body, IntSequence verbatim, int close) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        ids.addAll(tokenizer.encode(header));
        ids.add(channel);
        ids.addAll(tokenizer.encode(channelName));
        ids.add(message);
        if (verbatim != null) ids.addAll(verbatim);
        else ids.addAll(tokenizer.encode(body));
        ids.add(close);
        return Batch.prefill(ids.build().toArray());
    }

    /** The fixed Harmony system preamble: {@code <|start|>system<|message|>{...}<|end|>}. */
    @Override
    public List<Batch> conversationStart() {
        return conversationStart;
    }

    @Override
    public List<Batch> encodeTurn(Message m) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        if (m.role().equals(Role.SYSTEM)) { // conversation system -> developer block
            ids.addAll(tokenizer.encode("developer"));
            ids.add(message);
            ids.addAll(tokenizer.encode("# Instructions\n\n" + m.textOnly()));
        } else if (m.role().equals(Role.ASSISTANT)) { // history keeps only the final channel
            ids.addAll(tokenizer.encode("assistant"));
            ids.add(channel);
            ids.addAll(tokenizer.encode("final"));
            ids.add(message);
            ids.addAll(tokenizer.encode(m.textOnly()));
        } else {
            ids.addAll(tokenizer.encode(m.role().name()));
            ids.add(message);
            ids.addAll(tokenizer.encode(m.textOnly()));
        }
        ids.add(end);
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    /** {@code <|start|>assistant} - the model emits its own channel tokens from here. */
    @Override
    public List<Batch> generationPrompt(boolean thinking) {
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(start);
        ids.addAll(tokenizer.encode("assistant"));
        return List.of(Batch.prefill(ids.build().toArray()));
    }

    /** Closes the open message: {@code <|end|>} (the {@code <|return|>} stop is never ingested). */
    @Override
    public List<Batch> closeTurn() {
        return List.of(Batch.prefill(new int[] {end}));
    }

    /**
     * The codec face, tools included - the template's whole-conversation flow: the system preamble
     * grows the commentary-routing line when tools are offered; the developer block carries {@code
     * # Instructions} and/or the {@code # Tools} TypeScript namespace; an assistant call turn is
     * {@code <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{args}
     * <|call|>}, preceded by its analysis message only while no later final answer exists; a tool
     * result is {@code <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>
     * {content}<|end|>}, its name resolved from the most recent call (the template's max-one-call-
     * per-message assumption; extra calls in one turn are not rendered, matching it). Reasoning on
     * a plain assistant turn is dropped from history (the template's stated CoT rule).
     */
    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        List<Tool> tools = conversation.tools();
        TurnTemplate.requireToolShapes(msgs);
        List<Batch> out = new ArrayList<>();
        if (tools.isEmpty()) out.addAll(conversationStart());
        else out.add(toolsPreamble);
        // the answered-round-trip boundary: calls at or after it keep their analysis message
        int lastFinal = -1;
        for (int i = 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.ASSISTANT)
                    && m.content().stream().noneMatch(p -> p instanceof Part.ToolCall)) {
                lastFinal = i;
            }
        }
        Message dev =
                !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM) ? msgs.get(0) : null;
        String devText = dev == null ? "" : dev.textOnly();
        if (!devText.isEmpty() || !tools.isEmpty()) {
            StringBuilder body = new StringBuilder();
            if (!devText.isEmpty()) body.append("# Instructions\n\n").append(devText);
            if (!tools.isEmpty()) {
                if (!devText.isEmpty()) body.append("\n\n");
                body.append("# Tools\n\n").append(GptOssToolSyntax.namespace(tools));
            }
            out.add(block("developer", body.toString()));
        }
        String lastCall = null;
        for (int i = dev == null ? 0 : 1; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.TOOL)) {
                if (lastCall == null)
                    throw new UnsupportedConversation("tool result without a preceding call");
                StringBuilder body = new StringBuilder();
                for (Part p : m.content()) {
                    if (p instanceof Part.ToolResult r) body.append(r.text());
                }
                out.add(
                        turn(
                                "functions." + lastCall + " to=assistant",
                                "commentary",
                                body.toString(),
                                null,
                                end));
                continue;
            }
            if (!m.role().equals(Role.ASSISTANT)) {
                out.addAll(encodeTurn(m));
                continue;
            }
            Part.ToolCall first = null;
            for (Part p : m.content()) {
                if (p instanceof Part.ToolCall c) {
                    first = c;
                    break;
                }
            }
            if (first == null) { // history keeps only the final channel; CoT is dropped
                out.addAll(encodeTurn(new Message(Role.ASSISTANT, m.text())));
                lastCall = null;
                continue;
            }
            String content = m.text();
            Part.Reasoning thinking = m.reasoning();
            if (!content.isEmpty() && thinking != null)
                throw new UnsupportedConversation(
                        "assistant call turn with both content and thinking");
            if ((!content.isEmpty() || thinking != null) && i > lastFinal) {
                // the one place inference retains CoT: an unanswered tool-call round-trip
                String analysis = !content.isEmpty() ? content : thinking.text();
                out.add(
                        turn(
                                "assistant",
                                "analysis",
                                analysis,
                                thinking == null ? null : thinking.verbatim(),
                                end));
            }
            out.add(
                    turn(
                            "assistant to=functions." + first.name(),
                            "commentary json",
                            ToolCallSyntax.jinjaJson(first.arguments()),
                            first.verbatim(),
                            call));
            lastCall = first.name();
        }
        out.addAll(generationPrompt(conversation.thinking()));
        return out;
    }

    /**
     * Forced calls: the reply is seeded with {@code <|channel|>}, then the pin walks the header's
     * plain bytes {@code commentary to=functions.} through an offered name and releases - the
     * content-type, arguments and {@code <|call|>} stay the model's own.
     */
    /** Forced calls seed {@code <|channel|>} - the reply opens in a header the pin then owns. */
    @Override
    public int[] callSeed() {
        return new int[] {channel};
    }

    @Override
    public Optional<String> callGrammar(List<Tool> tools) {
        if (tools.isEmpty()) return Optional.empty();
        return Optional.of(ToolCallSyntax.prefixPinGbnf("commentary to=functions.", tools));
    }

    /**
     * {@code " <|constrain|>json<|message|>"} - the header's remainder after the pinned name, as
     * the model emits it natively. Scaffold, so forced: sampling it from the pinned (off-policy)
     * state is warmup-noise fragile (observed: the model abandoning the call with {@code <|end|>}
     * and restarting its reply).
     */
    @Override
    public int[] callEpilogue() {
        return callEpilogue;
    }

    @Override
    public ReplyParser parser() {
        return new HarmonyReplyParser(tokenizer);
    }
}
