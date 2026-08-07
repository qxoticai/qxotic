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
import java.util.function.BiConsumer;
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
     * A CLI generation outcome: the raw result, the display text the parser assembled, and the
     * parser's structured reply message (verbatim ids - what a codec chat loop appends).
     */
    record Reply(Generator.GenerationResult result, String text, Message message) {}

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
                                    replaceControlCharacters(tokenizer.decode(new int[] {token})));
                            printer.accept(token);
                        };
        int startPosition = state.position();
        // GENERATED tokens; the state was sized for prompt + budget, so nothing is subtracted
        int budget = options.maxOutputTokens();
        int totalPrompt = promptTokens.length();
        ReplyParser parser = ReplyParser.spans(tokenizer);
        StringBuilder text = new StringBuilder();
        Thinking.Inline inlineThink = new Thinking.Inline();
        BiConsumer<String, Boolean> collect =
                (fragment, reasoning) -> {
                    if (!reasoning) {
                        text.append(
                                options.think() ? inlineThink.project(fragment, false) : fragment);
                    } else if (options.think()) {
                        // thinking shown: bracket it inline in the display text (the old
                        // visible-tokens rendering kept think spans when --think is on)
                        text.append(inlineThink.project(fragment, true));
                    }
                };
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
                            onToken.accept(token);
                            String fragment = parser.feed(token);
                            if (!fragment.isEmpty()) collect.accept(fragment, parser.reasoning());
                            return true;
                        },
                        afterIngest);
        Message message = parser.finish();
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
        if (!options.stream()) {
            // nothing streamed, so the reply prints once, whole - here, so no mode can forget
            System.out.println(text);
        }
        return new Reply(result, text.toString(), message);
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
