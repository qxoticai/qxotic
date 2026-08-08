package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Thinking;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.PrintStream;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * One generation pass plus its terminal presentation - the part every CLI mode shares, whichever of
 * them assembled the prompt: prompt echo, token streaming, inline-or-stderr thinking, and the
 * stderr timing summary. {@code Instruct} and {@code Chat} differ in where prompts come from and
 * what happens between turns; the turn itself is this, once.
 */
final class Turn {

    private Turn() {}

    private static final String ANSI_GREY = "\033[90m";
    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_RESET = "\033[0m";

    /**
     * A CLI generation outcome: the raw result, the display text the parser assembled, the parser's
     * structured reply message (verbatim ids - what a codec chat loop appends), and {@code
     * kvTokens} - the generated tokens the KV actually holds, which the chat loop must track
     * exactly (plain decode never ingests the final sampled token; speculation commits verified
     * tokens the emission may not include).
     */
    record Reply(
            Generator.GenerationResult result,
            String text,
            Message message,
            IntSequence kvTokens) {}

    static <S extends RuntimeState> Reply generate(
            LoadedModel<S> model,
            S state,
            IntSequence promptTokens,
            Set<Integer> stopTokens,
            Sampler sampler,
            Options options) {
        return generate(model, state, promptTokens, stopTokens, sampler, options, null);
    }

    /**
     * The terminal-facing half of a turn, built once and shared by the plain and speculative paths:
     * prompt echo, the streaming printer, the span parser, and the think-aware display text. {@link
     * #sink()} is the one per-token entry both decode loops feed.
     */
    private static final class Rendering {
        final ReplyParser parser;
        final StringBuilder text = new StringBuilder();
        private final IntConsumer sink;

        Rendering(LoadedModel<?> model, IntSequence promptTokens, Options options) {
            Tokenizer tokenizer = model.tokenizer();
            if (options.echo()) {
                echoPrompt(tokenizer, promptTokens);
            }
            IntConsumer printer = streamingPrinter(tokenizer, options);
            IntConsumer onToken =
                    !options.echo()
                            ? printer
                            : token -> {
                                System.err.print(
                                        replaceControlCharacters(
                                                tokenizer.decode(new int[] {token})));
                                printer.accept(token);
                            };
            this.parser = ReplyParser.spans(tokenizer);
            Thinking.Inline inlineThink = new Thinking.Inline();
            this.sink =
                    token -> {
                        onToken.accept(token);
                        String fragment = parser.feed(token);
                        if (fragment.isEmpty()) {
                            return;
                        }
                        if (!parser.reasoning()) {
                            text.append(
                                    options.think()
                                            ? inlineThink.project(fragment, false)
                                            : fragment);
                        } else if (options.think()) {
                            // thinking shown: bracket it inline in the display text
                            text.append(inlineThink.project(fragment, true));
                        }
                    };
        }

        IntConsumer sink() {
            return sink;
        }

        /** The whole reply at once when nothing streamed - here, so no mode can forget. */
        void printUnlessStreamed(Options options) {
            if (!options.stream()) {
                System.out.println(text);
            }
        }
    }

    /**
     * As {@link #generate(LoadedModel, RuntimeState, IntSequence, Set, Sampler, Options)} with a
     * per-token ingest callback, which the prompt-cache path uses to append the tail.
     */
    static <S extends RuntimeState> Reply generate(
            LoadedModel<S> model,
            S state,
            IntSequence promptTokens,
            Set<Integer> stopTokens,
            Sampler sampler,
            Options options,
            IntConsumer afterIngest) {
        if (afterIngest == null
                && model.model() instanceof SpeculativeDecoding<?> speculative
                && speculative.speculationReady()) {
            // the prompt-cache path (afterIngest) tracks per-token ingests, which speculation's
            // batched commit-and-rollback does not produce - it keeps the plain loop
            @SuppressWarnings("unchecked")
            SpeculativeDecoding<S> capable = (SpeculativeDecoding<S>) speculative;
            return speculate(model, state, promptTokens, stopTokens, sampler, options, capable);
        }
        Rendering rendering = new Rendering(model, promptTokens, options);
        int startPosition = state.position();
        // GENERATED tokens; the state was sized for prompt + budget, so nothing is subtracted
        int budget = options.maxOutputTokens();
        int totalPrompt = promptTokens.length();
        Generator.GenerationResult result =
                Generator.generate(
                        model.model(),
                        state,
                        promptTokens.isEmpty()
                                ? List.of()
                                : List.of(Batch.prefill(promptTokens.toArray())),
                        sampler,
                        budget,
                        0 /* CLI: no deadline */,
                        stopTokens,
                        token -> {
                            rendering.sink().accept(token);
                            return true;
                        },
                        afterIngest);
        Message message = rendering.parser.finish();
        int generated = result.tokens().length() + (result.stopToken() >= 0 ? 1 : 0);
        String timingPrefix = options.colors() ? ANSI_CYAN : "";
        String timingSuffix = options.colors() ? ANSI_RESET : "";
        System.err.printf(
                "%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%s%n",
                timingPrefix,
                startPosition + totalPrompt + generated,
                state.contextCapacity(),
                totalPrompt / (result.promptNanos() / 1e9),
                totalPrompt,
                generated / (result.predictedNanos() / 1e9),
                generated,
                timingSuffix);
        rendering.printUnlessStreamed(options);
        // the plain loop never ingests the final sampled token: on a stop that token was the
        // stop itself (not in tokens), otherwise it is tokens' last element
        IntSequence kv = result.tokens();
        if (result.stopToken() < 0 && !kv.isEmpty()) {
            kv = kv.subSequence(0, kv.length() - 1);
        }
        return new Reply(result, rendering.text.toString(), message, kv);
    }

    /**
     * The speculative turn: same prompt echo, streaming, parsing and timing as the plain path, with
     * the decode loop replaced by the model's own draft-and-verify ({@link
     * SpeculativeDecoding#speculate}) and one extra stderr line reporting whether it paid.
     */
    private static <S extends RuntimeState> Reply speculate(
            LoadedModel<S> model,
            S state,
            IntSequence promptTokens,
            Set<Integer> stopTokens,
            Sampler sampler,
            Options options,
            SpeculativeDecoding<S> speculative) {
        Rendering rendering = new Rendering(model, promptTokens, options);
        int totalPrompt = promptTokens.length();
        // prompt ingest through the ordinary path (budget 0 = prefill only, logits retained)
        Generator.GenerationResult prefill =
                Generator.generate(
                        model.model(),
                        state,
                        promptTokens.isEmpty()
                                ? List.of()
                                : List.of(Batch.prefill(promptTokens.toArray())),
                        sampler,
                        0,
                        0,
                        stopTokens,
                        token -> true);
        long decodeStart = System.nanoTime();
        SpeculativeDecoding.Speculation spec =
                speculative.speculate(
                        state,
                        options.maxOutputTokens(),
                        0 /* CLI: no deadline */,
                        stopTokens,
                        sampler,
                        options.specDepth(),
                        token -> {
                            rendering.sink().accept(token);
                            return true;
                        });
        long decodeNanos = System.nanoTime() - decodeStart;
        Message message = rendering.parser.finish();
        Generator.GenerationResult result =
                new Generator.GenerationResult(
                        spec.emitted(),
                        spec.stopToken(),
                        spec.stopToken() >= 0 ? "stop" : "length",
                        prefill.promptNanos(),
                        decodeNanos);
        int generated = spec.emitted().length() + (spec.stopToken() >= 0 ? 1 : 0);
        String timingPrefix = options.colors() ? ANSI_CYAN : "";
        String timingSuffix = options.colors() ? ANSI_RESET : "";
        System.err.printf(
                "%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)"
                        + " speculation: depth %d, %.2f tokens/forward, %d%% drafts accepted%s%n",
                timingPrefix,
                state.position(),
                state.contextCapacity(),
                totalPrompt / (prefill.promptNanos() / 1e9),
                totalPrompt,
                generated / (decodeNanos / 1e9),
                generated,
                options.specDepth(),
                spec.forwards() == 0 ? 0.0 : (double) spec.committed().length() / spec.forwards(),
                spec.drafted() == 0 ? 0 : Math.round(100.0 * spec.accepted() / spec.drafted()),
                timingSuffix);
        rendering.printUnlessStreamed(options);
        return new Reply(result, rendering.text.toString(), message, spec.committed());
    }

    private static IntConsumer streamingPrinter(Tokenizer tokenizer, Options options) {
        if (!options.stream()) {
            return token -> {};
        }

        var open = SpecialTokens.find(tokenizer, "<think>");
        var close = SpecialTokens.find(tokenizer, "</think>");
        if (open.isEmpty() || close.isEmpty()) {
            return token -> { // no think markers in the vocabulary: plain content streaming
                if (!SpecialTokens.isSpecial(tokenizer, token)) {
                    byte[] bytes = tokenizer.decodeBytes(new int[] {token});
                    System.out.write(bytes, 0, bytes.length);
                }
            };
        }

        int thinkOpen = open.getAsInt();
        int thinkClose = close.getAsInt();
        boolean thinkEnabled = options.think();
        PrintStream thoughtOut = options.thinkInline() ? System.out : System.err;
        boolean ansi = options.colors();
        boolean[] inThink = {false};
        boolean[] emitted = {false};
        return token -> {
            if (token == thinkOpen) {
                if (thinkEnabled) {
                    onThinkingStart(thoughtOut, ansi);
                }
                inThink[0] = true;
                emitted[0] = false;
                return;
            }
            if (token == thinkClose) {
                if (thinkEnabled) {
                    onThinkingEnd(thoughtOut, ansi, emitted[0]);
                }
                inThink[0] = false;
                emitted[0] = false;
                return;
            }
            if (!SpecialTokens.isSpecial(tokenizer, token)) {
                byte[] bytes = tokenizer.decodeBytes(new int[] {token});
                if (inThink[0]) {
                    if (thinkEnabled) {
                        thoughtOut.write(bytes, 0, bytes.length);
                        emitted[0] = true;
                    }
                } else {
                    System.out.write(bytes, 0, bytes.length);
                }
            }
        };
    }

    private static void onThinkingStart(PrintStream thoughtOut, boolean ansi) {
        if (ansi) {
            thoughtOut.print(ANSI_GREY);
        }
        thoughtOut.println("[Start thinking]");
    }

    private static void onThinkingEnd(PrintStream thoughtOut, boolean ansi, boolean emitted) {
        if (emitted) {
            thoughtOut.println();
        }
        thoughtOut.println("[End thinking]");
        if (ansi) {
            thoughtOut.print(ANSI_RESET);
        }
        thoughtOut.println();
    }

    /** {@code --echo}: the prompt tokens to stderr, control characters escaped. */
    static void echoPrompt(Tokenizer tokenizer, IntSequence promptTokens) {
        promptTokens.forEachInt(
                token ->
                        System.err.print(
                                replaceControlCharacters(tokenizer.decode(new int[] {token}))));
    }

    /** Escape control characters (except newline) so token echo cannot distort the terminal. */
    private static String replaceControlCharacters(String str) {
        StringBuilder chars = new StringBuilder();
        str.codePoints()
                .forEach(
                        cp -> {
                            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
                            } else {
                                chars.appendCodePoint(cp);
                            }
                        });
        return chars.toString();
    }
}
