package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Hand-written SmolLM3 chat framing (ChatML dialect with a metadata system header), matching the
 * GGUF chat_template. The header renders ALWAYS: {@code ## Metadata} (knowledge cutoff, today's
 * date via {@code strftime_now("%d %B %Y")} - a constructor argument so tests are deterministic -
 * and the reasoning mode) then {@code ## Custom Instructions} (the system message with the {@code
 * /think} / {@code /no_think} switches stripped, or the mode's default persona), then {@code ###
 * Tools} when tools are offered: PYTHON-REPR tool signatures ({@code tool | string} in the
 * template) inside {@code <tools>} text, and the {@code <tool_call>} JSON instruction whose marker
 * spellings emit as trusted ids. Faithful quirk: without tools the template never closes the system
 * turn ({@code <|im_end|>} lives inside its tools branch).
 *
 * <p>A system message containing {@code /system_override} replaces the whole header with itself.
 * {@code /no_think} prefixes assistant turns and the generation prompt with the empty {@code
 * <think>} pair. Tool RESULTS render as plain user turns (the template's {@code tool} branch).
 *
 * <p>Deviation (documented): the template ignores {@code tool_calls} on history messages entirely;
 * this port renders them as the model emits them - {@code <tool_call>\n{"name": ..., "arguments":
 * ...}\n</tool_call>} with trusted marker ids - so call context survives the echo.
 */
public final class SmolLm3ChatTemplate implements ChatTemplate {

    // The think persona, split where its literal <think> spellings sit (specials in this vocab:
    // the render+rescan mints ids there, so the port emits them as trusted ids)
    static final String PERSONA_THINK_HEAD =
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an"
                + " assistant involves thoroughly exploring questions through a systematic thinking"
                + " process before providing the final precise and accurate solutions. This"
                + " requires engaging in a comprehensive cycle of analysis, summarizing,"
                + " exploration, reassessment, reflection, backtracking, and iteration to develop"
                + " well-considered thinking process. Please structure your response into two main"
                + " sections: Thought and Solution using the specified format: ";
    static final String PERSONA_THINK_MID = " Thought section ";
    static final String PERSONA_THINK_TAIL =
            " Solution section. In the Thought section, detail your reasoning process in steps."
                + " Each step should include detailed considerations such as analysing questions,"
                + " summarizing relevant findings, brainstorming new ideas, verifying the accuracy"
                + " of the current steps, refining any errors, and revisiting previous steps. In"
                + " the Solution section, based on various attempts, explorations, and reflections"
                + " from the Thought section, systematically present the final solution that you"
                + " deem correct. The Solution section should be logical, accurate, and concise and"
                + " detail necessary steps needed to reach the conclusion.\n\n";
    static final String DEFAULT_THINK_PERSONA =
            PERSONA_THINK_HEAD + "<think>" + PERSONA_THINK_MID + "</think>" + PERSONA_THINK_TAIL;
    static final String DEFAULT_PERSONA =
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n";
    static final String TOOLS_INTRO =
            "You may call one or more functions to assist with the user query.\nYou are provided"
                    + " with function signatures within <tools></tools> XML tags:\n\n<tools>\n";
    static final String TOOLS_OUTRO_HEAD =
            "</tools>\n\nFor each function call, return a json object with function name and"
                    + " arguments within ";
    static final String TOOLS_EXAMPLE_BODY =
            "\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n";

    private final Tokenizer tokenizer;
    private final String today; // strftime_now("%d %B %Y")
    private final int imStart; // <|im_start|>
    private final int imEnd; // <|im_end|>
    private final int think; // <think>
    private final int endThink; // </think>
    private final int toolCall; // <tool_call>
    private final int endToolCall; // </tool_call>

    public SmolLm3ChatTemplate(Tokenizer tokenizer) {
        this(
                tokenizer,
                LocalDate.now()
                        .format(DateTimeFormatter.ofPattern("dd MMMM yyyy", Locale.ENGLISH)));
    }

    public SmolLm3ChatTemplate(Tokenizer tokenizer, String today) {
        this.tokenizer = tokenizer;
        this.today = today;
        this.imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        this.imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        this.think = SpecialTokens.require(tokenizer, "<think>");
        this.endThink = SpecialTokens.require(tokenizer, "</think>");
        this.toolCall = SpecialTokens.require(tokenizer, "<tool_call>");
        this.endToolCall = SpecialTokens.require(tokenizer, "</tool_call>");
    }

    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        TurnTemplate.requireToolShapes(msgs);
        List<Tool> tools = conversation.tools();

        Message sys =
                !msgs.isEmpty() && msgs.get(0).role().equals(Role.SYSTEM) ? msgs.get(0) : null;
        String sysContent = sys == null ? null : sys.textOnly();
        boolean thinkMode = conversation.thinking();
        String custom = "";
        if (sysContent != null) {
            if (sysContent.contains("/no_think")) thinkMode = false;
            else if (sysContent.contains("/think")) thinkMode = true;
            custom = sysContent.replace("/no_think", "").replace("/think", "").stripTrailing();
        }

        IntSequence.Builder ids = IntSequence.newBuilder();
        StringBuilder text = new StringBuilder();
        ids.add(imStart);
        text.append("system\n");
        if (sysContent != null && sysContent.contains("/system_override")) {
            text.append(custom.replace("/system_override", "").stripTrailing());
            emit(ids, text, imEnd);
            text.append('\n');
        } else {
            text.append("## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: ")
                    .append(today)
                    .append("\nReasoning Mode: ")
                    .append(thinkMode ? "/think" : "/no_think")
                    .append("\n\n## Custom Instructions\n\n");
            if (!custom.isEmpty()) {
                text.append(custom).append("\n\n");
            } else if (thinkMode) {
                text.append(PERSONA_THINK_HEAD);
                emit(ids, text, think);
                text.append(PERSONA_THINK_MID);
                emit(ids, text, endThink);
                text.append(PERSONA_THINK_TAIL);
            } else {
                text.append(DEFAULT_PERSONA);
            }
            if (!tools.isEmpty()) {
                text.append("### Tools\n\n").append(TOOLS_INTRO);
                for (Tool t : tools) {
                    text.append(ToolCallSyntax.pythonRepr(JsonCodec.parse(t.rawJson())))
                            .append('\n');
                }
                text.append(TOOLS_OUTRO_HEAD);
                emit(ids, text, toolCall);
                emit(ids, text, endToolCall);
                text.append(" XML tags:\n");
                emit(ids, text, toolCall);
                text.append(TOOLS_EXAMPLE_BODY);
                emit(ids, text, endToolCall);
                text.append("\n\n");
                emit(ids, text, imEnd);
                text.append('\n');
            }
            // faithful quirk: without tools the template leaves the system turn UNCLOSED
        }

        for (int i = sys != null ? 1 : 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.ASSISTANT)) {
                emit(ids, text, imStart);
                text.append("assistant\n");
                if (!thinkMode) {
                    emit(ids, text, think);
                    text.append("\n\n");
                    emit(ids, text, endThink);
                    text.append('\n');
                }
                String content = stripLeadingNewlines(m.text()); // Reasoning dropped from history
                text.append(content);
                boolean first = true;
                for (Part p : m.content()) {
                    if (!(p instanceof Part.ToolCall call)) continue;
                    if (!first || !content.isEmpty()) text.append('\n');
                    first = false;
                    emit(ids, text, toolCall);
                    text.append("\n{\"name\": \"")
                            .append(call.name())
                            .append("\", \"arguments\": ")
                            .append(ToolCallSyntax.jinjaJson(call.arguments()))
                            .append("}\n");
                    emit(ids, text, endToolCall);
                }
                emit(ids, text, imEnd);
                text.append('\n');
            } else { // user, later system, and tool results all frame as user-family turns
                emit(ids, text, imStart);
                text.append(m.role().equals(Role.TOOL) ? "user" : m.role().name()).append('\n');
                if (m.role().equals(Role.TOOL)) {
                    for (Part p : m.content()) {
                        if (p instanceof Part.ToolResult r) text.append(r.text());
                    }
                } else {
                    text.append(m.textOnly());
                }
                emit(ids, text, imEnd);
                text.append('\n');
            }
        }

        emit(ids, text, imStart);
        text.append("assistant\n");
        if (!thinkMode) {
            emit(ids, text, think);
            text.append("\n\n");
            emit(ids, text, endThink);
            text.append('\n');
        }
        flush(ids, text);
        List<Batch> out = new ArrayList<>();
        out.add(Batch.prefill(ids.build().toArray()));
        return out;
    }

    private static String stripLeadingNewlines(String s) {
        int i = 0;
        while (i < s.length() && s.charAt(i) == '\n') i++;
        return s.substring(i);
    }

    private void emit(IntSequence.Builder ids, StringBuilder text, int id) {
        flush(ids, text);
        ids.add(id);
    }

    private void flush(IntSequence.Builder ids, StringBuilder text) {
        if (text.isEmpty()) return;
        ids.addAll(tokenizer.encode(text.toString()));
        text.setLength(0);
    }

    /** Tool calls parse from the {@code <tool_call>} span's JSON {@code {name, arguments}}. */
    @Override
    public ReplyParser parser() {
        return ReplyParser.spans(
                tokenizer, "<tool_call>", "</tool_call>", ToolCallSyntax::parseBlock);
    }

    /** Forced calls seed {@code <tool_call>} (seeding only - no pin hook yet). */
    @Override
    public int[] callSeed() {
        return new int[] {toolCall};
    }

    /** No-think prompts close the empty pair in the prompt: pre-feed it. */
    @Override
    public int[] replySeed(boolean thinking) {
        if (thinking) return new int[0]; // the model emits its own <think>
        IntSequence.Builder ids = IntSequence.newBuilder();
        ids.add(think);
        ids.addAll(tokenizer.encode("\n\n"));
        ids.add(endThink);
        ids.addAll(tokenizer.encode("\n"));
        return ids.build().toArray();
    }
}
