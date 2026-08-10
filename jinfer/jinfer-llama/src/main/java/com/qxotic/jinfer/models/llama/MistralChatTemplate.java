package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.Tokenizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Hand-written Ministral 3 (Mistral v13 wire) chat framing: {@code <s>} bos, {@code
 * [SYSTEM_PROMPT]...[/SYSTEM_PROMPT]} (the template's long default persona - literal {@code
 * {today}} placeholders included - when the conversation has no system turn), {@code
 * [AVAILABLE_TOOLS]} + the WHOLE tool list as one JSON array, {@code [INST]...[/INST]} user turns,
 * bare assistant continuations closed by {@code </s>}, calls as {@code
 * [TOOL_CALLS]name[ARGS]{args}} (no close marker - the next call or {@code </s>} ends one), and
 * {@code [TOOL_RESULTS]...[/TOOL_RESULTS]} results. No generation-prompt scaffold: the assistant
 * continues directly after {@code [/INST]}. No thinking scaffold ({@link Part.Reasoning} on history
 * is dropped).
 */
public final class MistralChatTemplate implements ChatTemplate {

    static final String DEFAULT_SYSTEM =
            "You are Ministral-3-3B-Instruct-2512, a Large Language Model (LLM) created by Mistral"
                + " AI, a French startup headquartered in Paris.\n"
                + "You power an AI assistant called Le Chat.\n"
                + "Your knowledge base was last updated on 2023-10-01.\n"
                + "The current date is {today}.\n"
                + "\n"
                + "When you're not sure about some information or when the user's request requires"
                + " up-to-date or specific data, you must use the available tools to fetch the"
                + " information. Do not hesitate to use tools whenever they can provide a more"
                + " accurate or complete response. If no relevant tools are available, then clearly"
                + " state that you don't have the information and avoid making up anything.\n"
                + "If the user's question is not clear, ambiguous, or does not provide enough"
                + " context for you to accurately answer the question, you do not try to answer it"
                + " right away and you rather ask the user to clarify their request (e.g. \"What"
                + " are some good restaurants around me?\" => \"Where are you?\" or \"When is the"
                + " next flight to Tokyo\" => \"Where do you travel from?\").\n"
                + "You are always very attentive to dates, in particular you try to resolve dates"
                + " (e.g. \"yesterday\" is {yesterday}) and when asked about information at"
                + " specific dates, you discard information that is at another date.\n"
                + "You follow these instructions in all languages, and always respond to the user"
                + " in the language they use or request.\n"
                + "Next sections describe the capabilities that you have.\n"
                + "\n"
                + "# WEB BROWSING INSTRUCTIONS\n"
                + "\n"
                + "You cannot perform any web search or access internet to open URLs, links etc. If"
                + " it seems like the user is expecting you to do so, you clarify the situation and"
                + " ask the user to copy paste the text directly in the chat.\n"
                + "\n"
                + "# MULTI-MODAL INSTRUCTIONS\n"
                + "\n"
                + "You have the ability to read images, but you cannot generate images. You also"
                + " cannot transcribe audio files or videos.\n"
                + "You cannot read nor transcribe audio files or videos.\n"
                + "\n"
                + "# TOOL CALLING INSTRUCTIONS\n"
                + "\n"
                + "You may have access to tools that you can use to fetch information or perform"
                + " actions. You must use these tools in the following situations:\n"
                + "\n"
                + "1. When the request requires up-to-date information.\n"
                + "2. When the request requires specific data that you do not have in your"
                + " knowledge base.\n"
                + "3. When the request involves actions that you cannot perform without tools.\n"
                + "\n"
                + "Always prioritize using tools to provide the most accurate and helpful response."
                + " If tools are not available, inform the user that you cannot perform the"
                + " requested action at the moment.";

    private final Tokenizer tokenizer;
    private final int bos; // <s>
    private final int eos; // </s>
    private final int inst; // [INST]
    private final int endInst; // [/INST]
    private final int systemPrompt; // [SYSTEM_PROMPT]
    private final int endSystemPrompt; // [/SYSTEM_PROMPT]
    private final int availableTools; // [AVAILABLE_TOOLS]
    private final int endAvailableTools; // [/AVAILABLE_TOOLS]
    private final int toolCalls; // [TOOL_CALLS]
    private final int args; // [ARGS]
    private final int toolResults; // [TOOL_RESULTS]
    private final int endToolResults; // [/TOOL_RESULTS]
    private final TokenRuns proto; // compiled spelling table, forked per encode

    public MistralChatTemplate(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.bos = SpecialTokens.require(tokenizer, "<s>");
        this.eos = SpecialTokens.require(tokenizer, "</s>");
        this.inst = SpecialTokens.require(tokenizer, "[INST]");
        this.endInst = SpecialTokens.require(tokenizer, "[/INST]");
        this.systemPrompt = SpecialTokens.require(tokenizer, "[SYSTEM_PROMPT]");
        this.endSystemPrompt = SpecialTokens.require(tokenizer, "[/SYSTEM_PROMPT]");
        this.availableTools = SpecialTokens.require(tokenizer, "[AVAILABLE_TOOLS]");
        this.endAvailableTools = SpecialTokens.require(tokenizer, "[/AVAILABLE_TOOLS]");
        this.toolCalls = SpecialTokens.require(tokenizer, "[TOOL_CALLS]");
        this.args = SpecialTokens.require(tokenizer, "[ARGS]");
        this.toolResults = SpecialTokens.require(tokenizer, "[TOOL_RESULTS]");
        this.endToolResults = SpecialTokens.require(tokenizer, "[/TOOL_RESULTS]");
        this.proto = new TokenRuns(tokenizer);
    }

    @Override
    public List<Batch> encode(Conversation conversation) {
        List<Message> msgs = conversation.messages();
        TurnTemplate.requireToolShapes(msgs);
        List<Tool> tools = conversation.tools();

        Message sys = Message.leadingSystem(msgs);
        TokenRuns runs = proto.fresh();
        runs.id(bos).id(systemPrompt);
        runs.text(sys != null ? sys.textOnly() : DEFAULT_SYSTEM);
        runs.id(endSystemPrompt);
        if (!tools.isEmpty()) {
            List<Object> parsed = new ArrayList<>();
            for (Tool t : tools) parsed.add(JsonCodec.parse(t.rawJson()));
            runs.id(availableTools).text(ToolCallSyntax.jinjaJson(parsed)).id(endAvailableTools);
        }
        for (int i = sys != null ? 1 : 0; i < msgs.size(); i++) {
            Message m = msgs.get(i);
            if (m.role().equals(Role.USER)) {
                runs.id(inst).text(m.textOnly()).id(endInst);
            } else if (m.role().equals(Role.TOOL)) {
                runs.id(toolResults);
                for (Part p : m.content()) {
                    if (p instanceof Part.ToolResult r) runs.text(r.text());
                }
                runs.id(endToolResults);
            } else { // assistant: bare content, calls, then the per-message eos
                runs.text(m.text()); // Reasoning dropped: no think scaffold in this wire
                for (Part p : m.content()) {
                    if (!(p instanceof Part.ToolCall call)) continue;
                    runs.id(toolCalls).text(call.name()).id(args);
                    runs.text(ToolCallSyntax.jinjaJson(call.arguments()));
                }
                runs.id(eos);
            }
        }
        // no generation scaffold: the assistant continues after [/INST]
        return List.of(runs.batch());
    }

    private ReplyLanguage.Selection autoReply; // memoized: tools-independent, built once

    /**
     * The reply-language walk: a call is {@code %[TOOL_CALLS] name %[ARGS] args-json}, the name and
     * args as free holes around the interior mark, close-LESS - the region exits when the next call
     * opens, the reply ends, or the payload's balance completes. The old self-closing span
     * heuristic derives from the shape.
     */
    @Override
    public ReplyParser parser() {
        if (autoReply == null) {
            autoReply = ReplyLanguage.Selection.of(language(ReplyLanguage.free()), tokenizer);
        }
        return autoReply.walk();
    }

    @Override
    public Optional<ReplyLanguage.Node> autoLanguage(ReplyLanguage.Node contentHole) {
        return Optional.of(language(contentHole));
    }

    /** The family tree with the content hole stated: {@code (content | call)* </s>?}. */
    private static ReplyLanguage.Node language(ReplyLanguage.Node contentHole) {
        return ReplyLanguage.seq(
                ReplyLanguage.rep(
                        ReplyLanguage.alt(
                                ReplyLanguage.content(contentHole),
                                ReplyLanguage.call(
                                        MistralChatTemplate::walkCalls,
                                        ReplyLanguage.mark("[TOOL_CALLS]"),
                                        ReplyLanguage.free(),
                                        ReplyLanguage.mark("[ARGS]"),
                                        ReplyLanguage.free())),
                        0,
                        -1),
                ReplyLanguage.opt(ReplyLanguage.mark("</s>")));
    }

    /**
     * The forced-call language this family NEVER had: the header is wholly forced ({@code
     * [TOOL_CALLS]name[ARGS]} - nothing is sampled until the arguments), the arguments are
     * schema-bound. The empty-prefix pin that dead-ended this model is structurally gone: there is
     * no free region between the seed and the name for the model to derail in.
     */
    @Override
    public Optional<ReplyLanguage.Node> forcedCallLanguage(List<Tool> tools) {
        if (tools.isEmpty()) return Optional.empty();
        List<ReplyLanguage.Node> options = new ArrayList<>(tools.size());
        for (Tool tool : tools) {
            options.add(
                    ReplyLanguage.call(
                            MistralChatTemplate::walkCalls,
                            ReplyLanguage.mark("[TOOL_CALLS]"),
                            ReplyLanguage.bytes(tool.name()),
                            ReplyLanguage.mark("[ARGS]"),
                            ReplyLanguage.gbnf(Grammar.schemaGbnf(tool.parameters()))));
        }
        return Optional.of(
                ReplyLanguage.seq(
                        new ReplyLanguage.Node.Alt(options),
                        ReplyLanguage.opt(ReplyLanguage.mark("</s>"))));
    }

    /**
     * The walk's payload is {@code name{json}} - the interior {@code [ARGS]} mark is excluded from
     * capture, so the name ends where the object begins. A payload without a parseable object is no
     * call.
     */
    static List<Part.ToolCall> walkCalls(String payload) {
        int brace = payload.indexOf('{');
        if (brace <= 0) return List.of();
        String name = payload.substring(0, brace).strip();
        if (name.isEmpty()) return List.of();
        try {
            if (JsonCodec.parse(payload.substring(brace)) instanceof Map<?, ?> parsed) {
                Map<String, Object> arguments = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : parsed.entrySet()) {
                    arguments.put(String.valueOf(e.getKey()), e.getValue());
                }
                return List.of(new Part.ToolCall("", name, arguments));
            }
        } catch (RuntimeException malformed) {
            // not a JSON object: no call
        }
        return List.of();
    }
}
