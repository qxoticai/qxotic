package com.qxotic.jinfer.models.lfm3;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.MediaEncodingCache;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.PromptWriter;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.SpokenTurn;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * LFM3 chat framing, token-exact with the GGUF's Jinja template (LFM3, LFM3-VL and LFM3-Omni share
 * it): {@code <|startoftext|>}, then {@code <|im_start|>{role}\n{content}<|im_end|>\n} per message
 * with content verbatim, and the generation prompt {@code <|im_start|>assistant\n} opening {@code
 * <think>} to reason or writing the closed {@code <think></think>} to answer directly.
 *
 * <ul>
 *   <li>Tools are declared in the system turn as one JSON object per line inside {@code
 *       <tools>...</tools>}, after the system prompt; calls are Qwen-style XML functions, {@code
 *       <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>},
 *       results a {@code tool} turn of {@code <tool_response>\n...\n</tool_response>}.
 *   <li>Assistant turns after the last user message keep their reasoning; earlier ones drop it and
 *       show an empty {@code <think></think>} before the answer.
 *   <li>Images (and Omni's audio) are {@code <|vision_start|> rows <|vision_end|>}, the rows taking
 *       the place of the template's {@code <|image|>} / {@code <|audio|>} placeholder.
 * </ul>
 */
final class Lfm3ChatTemplate implements ChatTemplate {

    private final Tokenizer tokenizer;
    private final Multimodal media;
    private final Lfm3Speech speech; // replays spoken turns; null without the Omni mmproj
    private final int bos, imStart, imEnd, think, endThink, visionOpen, visionClose;
    private final ReplyLanguage.Spans spans;
    private final Set<Integer> speechOpeners; // a text reply's forbidden mode switches

    private static final Pattern THINK_MARKER = Pattern.compile("</?think>");

    Lfm3ChatTemplate(Lfm3 model) {
        this(model.tokenizer(), model);
    }

    Lfm3ChatTemplate(Tokenizer tokenizer, Multimodal media) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.media = media;
        this.speech = media instanceof Lfm3 model ? model.speech() : null;
        bos = SpecialTokens.require(tokenizer, "<|startoftext|>");
        imStart = SpecialTokens.require(tokenizer, "<|im_start|>");
        imEnd = SpecialTokens.require(tokenizer, "<|im_end|>");
        think = SpecialTokens.require(tokenizer, "<think>");
        endThink = SpecialTokens.require(tokenizer, "</think>");
        visionOpen = SpecialTokens.require(tokenizer, "<|vision_start|>");
        visionClose = SpecialTokens.require(tokenizer, "<|vision_end|>");
        Set<Integer> openers = new HashSet<>();
        for (String opener : List.of("<|audio_start|>", "<|mixed_start|>"))
            SpecialTokens.find(tokenizer, opener).ifPresent(openers::add);
        speechOpeners = Set.copyOf(openers);
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

    @Override
    public ReplyState encode(Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
        return encode(conversation, batchCapacity, null, sink);
    }

    @Override
    public ReplyState encode(
            Conversation conversation,
            int batchCapacity,
            MediaEncodingCache mediaCache,
            Consumer<Batch> sink) {
        List<Message> messages = conversation.messages();
        if (messages.isEmpty()) throw new UnsupportedConversation("LFM3 needs a message");
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, mediaCache, sink);
        out.id(bos);

        int first = 0;
        String system = "";
        if (messages.get(0).role().equals(Role.SYSTEM)) {
            system = textOf(messages.get(0));
            first = 1;
        }
        List<Tool> declared =
                conversation.tools().stream().filter(tool -> !deferred(tool)).toList();
        if (!system.isEmpty() || !declared.isEmpty()) {
            out.id(imStart).text("system\n").text(system);
            if (!declared.isEmpty()) {
                out.text(system.isEmpty() ? "" : "\n\n").trusted("<tools>");
                for (Tool tool : declared) out.text("\n").text(toolJson(tool));
                out.text("\n").trusted("</tools>");
            }
            out.id(imEnd).text("\n");
            out.flush();
        }

        int lastUser = -1;
        for (int i = first; i < messages.size(); i++)
            if (messages.get(i).role().equals(Role.USER)) lastUser = i;
        for (int i = first; i < messages.size(); i++) {
            Message message = messages.get(i);
            out.id(imStart).text(message.role().name() + "\n");
            if (message.role().equals(Role.ASSISTANT)) writeAssistant(out, message, i > lastUser);
            else if (message.role().equals(Role.TOOL)) writeToolResult(out, message);
            else writeContent(out, message);
            out.id(imEnd).text("\n");
            out.flush();
        }

        out.id(imStart).text("assistant\n").id(think);
        IntSequence.Builder prefix = IntSequence.newBuilder().add(think);
        if (!conversation.thinking()) {
            out.id(endThink);
            prefix.add(endThink);
        }
        out.finish();
        ReplyParser parser = spans.parser();
        IntSequence replyPrefix = prefix.build();
        parser.seed(replyPrefix);
        return new ReplyState(replyPrefix, parser);
    }

    /** {@code parse_content}: text verbatim, media framed as the template's placeholders. */
    private void writeContent(PromptWriter out, Message message) {
        for (Content part : message.content()) {
            switch (part) {
                case Content.Text text -> out.text(text.text());
                case Content.Media value -> writeMedia(out, value);
                default ->
                        throw new UnsupportedConversation(
                                message.role().name()
                                        + " turn: "
                                        + part.getClass().getSimpleName());
            }
        }
    }

    private void writeMedia(PromptWriter out, Content.Media content) {
        out.cachedMedia(
                content.value(),
                content.contentKey(),
                encoded ->
                        encoded.id(visionOpen)
                                .media(
                                        content.value(),
                                        content.contentKey(),
                                        projector(content.value()),
                                        false)
                                .id(visionClose));
    }

    /** The tower for this media's modality (records: the class IS the modality). */
    @SuppressWarnings("unchecked")
    private <R extends Media> MediaProjector<R> projector(R value) {
        Optional<MediaProjector<R>> projector =
                media == null ? Optional.empty() : media.projector((Class<R>) value.getClass());
        return projector.orElseThrow(
                () ->
                        new UnsupportedConversation(
                                value.getClass().getSimpleName().toLowerCase()
                                        + " input needs the model's mmproj"));
    }

    @Override
    public int mediaPositions(Media value) {
        return projector(value).positions(value);
    }

    /**
     * Reasoning kept: {@code <think>reasoning</think>}, or an empty pair when the content carries
     * none of its own; reasoning dropped: the text after the content's last {@code </think>},
     * trimmed, behind an empty pair. Then the content and one XML function block per call.
     */
    private void writeAssistant(PromptWriter out, Message message, boolean keepThinking) {
        for (Content part : message.content()) {
            if (part instanceof Content.Speech spoken) {
                writeSpoken(out, spoken.turn());
                return; // its transcript is in the turn's own record
            }
        }
        String content = "";
        String reasoning = null;
        for (Content part : message.content()) {
            switch (part) {
                case Content.Text text -> content += text.text();
                case Content.Reasoning r -> reasoning = r.text();
                case Content.ToolCall ignored -> {}
                default ->
                        throw new UnsupportedConversation(
                                "assistant turn: " + part.getClass().getSimpleName());
            }
        }
        if (keepThinking && reasoning != null) {
            out.id(think).text(reasoning).id(endThink);
        } else if (keepThinking && !content.contains("</think>")) {
            out.id(think).id(endThink);
        } else if (!keepThinking) {
            int at = content.lastIndexOf("</think>");
            if (at >= 0) content = content.substring(at + "</think>".length()).strip();
            out.id(think).id(endThink);
        }
        writeAssistantText(out, content);
        boolean firstCall = true;
        for (Content part : message.content()) {
            if (!(part instanceof Content.ToolCall call)) continue;
            if (!firstCall) out.text("\n\n");
            firstCall = false;
            out.trusted("<tool_call>\n<function=").text(call.name()).text(">\n");
            for (Map.Entry<String, Object> argument : call.arguments().entrySet()) {
                out.trusted("<parameter=").text(argument.getKey()).text(">\n");
                out.text(value(argument.getValue())).text("\n").trusted("</parameter>\n");
            }
            out.trusted("</function>\n</tool_call>");
        }
    }

    /**
     * A spoken turn as the model produced it: the empty think span, then every position - text ids
     * as they were, each frame as its feedback row, keyed by its codes like the live row - so the
     * prompt cache recognizes the turn it holds.
     */
    private void writeSpoken(PromptWriter out, SpokenTurn turn) {
        if (speech == null)
            throw new UnsupportedConversation("a spoken turn replays through LFM3-Omni's mmproj");
        out.id(think).id(endThink);
        Lfm3Speech.Feedback rows = speech.new Feedback(MemoryAllocators.ofArena(Arena.ofAuto()));
        Iterator<int[]> frames = turn.frames().iterator();
        IntSequence positions = turn.positions();
        for (int i = 0; i < positions.length(); i++) {
            int id = positions.intAt(i);
            if (id != turn.placeholder()) {
                out.id(id);
                continue;
            }
            int[] codes = frames.next();
            out.batch(rows.batch(codes));
        }
    }

    /**
     * The assistant's own text, its {@code <think>} / {@code </think>} spellings as the markers.
     */
    private void writeAssistantText(PromptWriter out, String text) {
        Matcher marker = THINK_MARKER.matcher(text);
        int at = 0;
        while (marker.find()) {
            out.text(text.substring(at, marker.start()));
            out.id(marker.group().equals("<think>") ? think : endThink);
            at = marker.end();
        }
        out.text(text.substring(at));
    }

    /** {@code <tool_response>\n...\n</tool_response>}, unless the result already carries one. */
    private void writeToolResult(PromptWriter out, Message message) {
        StringBuilder result = new StringBuilder();
        for (Content part : message.content()) {
            if (!result.isEmpty()) result.append('\n');
            switch (part) {
                case Content.ToolResult r -> result.append(r.text());
                case Content.Text t -> result.append(t.text());
                default ->
                        throw new UnsupportedConversation(
                                "tool turn: " + part.getClass().getSimpleName());
            }
        }
        String text = result.toString();
        if (text.contains("<tool_response>")) {
            out.text(text);
            return;
        }
        out.trusted("<tool_response>\n").text(text).trusted("\n</tool_response>");
    }

    /** The system turn takes text only, verbatim. */
    private static String textOf(Message system) {
        StringBuilder text = new StringBuilder();
        for (Content part : system.content()) {
            if (!(part instanceof Content.Text t))
                throw new UnsupportedConversation("LFM3 system messages take text only");
            text.append(t.text());
        }
        return text.toString();
    }

    /** {@code render_value}: strings raw, JSON spellings for booleans and null, tojson for maps. */
    static String value(Object v) {
        if (v instanceof Boolean b) return b ? "true" : "false";
        if (v == null) return "null";
        return v instanceof String s ? s : ToolCallSyntax.jinjaJson(v);
    }

    /** {@code tool_to_json}: the function object minus {@code defer_loading}. */
    private static String toolJson(Tool tool) {
        Map<String, Object> fn = new LinkedHashMap<>(ToolCallSyntax.functionObject(tool));
        fn.remove("defer_loading");
        return ToolCallSyntax.jinjaJson(fn);
    }

    /** A deferred tool is revealed later by a tool result, never declared up front. */
    private static boolean deferred(Tool tool) {
        return Boolean.TRUE.equals(ToolCallSyntax.functionObject(tool).get("defer_loading"))
                || Boolean.TRUE.equals(tool.definition().get("defer_loading"));
    }

    /** The GGUF {@code tts} template's language table: a code names its language. */
    private static final Map<String, String> TTS_LANGUAGES =
            Map.ofEntries(
                    Map.entry("ar", "Arabic"),
                    Map.entry("da", "Danish"),
                    Map.entry("de", "German"),
                    Map.entry("el", "Greek"),
                    Map.entry("en", "American English"),
                    Map.entry("en-us", "American English"),
                    Map.entry("en-gb", "British English"),
                    Map.entry("fi", "Finnish"),
                    Map.entry("fr", "French"),
                    Map.entry("he", "Hebrew"),
                    Map.entry("it", "Italian"),
                    Map.entry("ja", "Japanese"),
                    Map.entry("ko", "Korean"),
                    Map.entry("nl", "Dutch"),
                    Map.entry("pl", "Polish"),
                    Map.entry("pt", "Portuguese"),
                    Map.entry("es", "Spanish"),
                    Map.entry("sv", "Swedish"),
                    Map.entry("tr", "Turkish"),
                    Map.entry("zh", "Chinese"));

    /**
     * The GGUF's {@code tts} system turn: {@code language} a code from its table or a language name
     * as LiquidAI writes them ("American English"), null = English; {@code voice} {@code female}
     * (null) or {@code male}.
     */
    @Override
    public Optional<String> speechInstruction(String language, String voice) {
        String code = language == null || language.isEmpty() ? "en" : language.toLowerCase();
        String name = TTS_LANGUAGES.getOrDefault(code, language);
        String v = voice == null ? "female" : voice.toLowerCase();
        if (!v.equals("female") && !v.equals("male"))
            throw new IllegalArgumentException(
                    "LFM3 speaks with a female or a male voice, not '" + voice + "'");
        return Optional.of("Perform TTS in " + name + " using a " + v + " voice.");
    }

    /**
     * Speech and interleaved replies open with their own token; a text reply forbids both, as
     * LiquidAI's {@code generation_mode="text"} and llama.cpp's text mode do. Under a voice-chat
     * system prompt Omni opens {@code <|mixed_start|>} almost surely.
     */
    @Override
    public Set<Integer> replyForbidden() {
        return speechOpeners;
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

    /**
     * A forced call writes its header as bytes. ponytail: {@code <function=} is then plain-encoded,
     * not its vocabulary token (a mark would drop its spelling from the parsed payload); the model
     * reads the same text, and every call it writes itself uses the token.
     */
    @Override
    public Optional<ReplyLanguage.Selection> forcedCall(List<Tool> callableTools) {
        if (callableTools.isEmpty()) return Optional.empty();
        return Optional.of(spans.forcedCall(callableTools, tool -> "\n<function=" + tool.name()));
    }
}
