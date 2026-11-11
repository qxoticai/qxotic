package com.qxotic.jinfer.models.gemmaembedding2;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.PromptWriter;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.models.gemma4.Gemma4;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

/** EmbeddingGemma's input template: explicit special-aware text, framing and media placement. */
final class GemmaEmbedding2Template {
    private static final int VIDEO_FRAME_TOKENS = 140;

    private final Tokenizer tokenizer;
    private final Specials specials;
    private final IntSequence prefix, suffix;
    private final String bosSpelling;
    private final Gemma4.Towers towers;

    GemmaEmbedding2Template(Tokenizer tokenizer, int bos, int eos, Gemma4.Towers towers) {
        this(tokenizer, Specials.of(tokenizer), IntSequence.of(bos), IntSequence.of(eos), towers);
    }

    private GemmaEmbedding2Template(
            Tokenizer tokenizer,
            Specials specials,
            IntSequence prefix,
            IntSequence suffix,
            Gemma4.Towers towers) {
        this.tokenizer = tokenizer;
        this.specials = specials;
        this.prefix = prefix;
        this.suffix = suffix;
        this.bosSpelling = tokenizer.vocabulary().token(prefix.intAt(0));
        this.towers = towers;
    }

    GemmaEmbedding2Template withMedia(Gemma4.Towers towers) {
        return new GemmaEmbedding2Template(tokenizer, specials, prefix, suffix, towers);
    }

    /**
     * The reference processor disables automatic BOS/EOS when the rendered text starts with BOS.
     */
    IntSequence encodeText(String text) {
        Objects.requireNonNull(text, "text");
        IntSequence tokens = encodeTextRun(text);
        return text.startsWith(bosSpelling)
                ? tokens
                : IntSequence.concatAll(prefix, tokens, suffix);
    }

    private IntSequence encodeTextRun(CharSequence text) {
        return specials.encode(tokenizer, text);
    }

    /**
     * Adjacent strings form one text run; a medium ends the run and keeps its original position.
     */
    void encodeParts(List<?> parts, int batchCapacity, Consumer<Batch> sink) {
        PromptWriter out = new PromptWriter(tokenizer, batchCapacity, sink);
        boolean frame = true;
        for (int i = 0; i < parts.size(); ) {
            int start = i;
            Object part = parts.get(i++);
            if (part instanceof String text) {
                StringBuilder run = new StringBuilder(text);
                while (i < parts.size() && parts.get(i) instanceof String next) {
                    run.append(next);
                    i++;
                }
                String joined = run.toString();
                if (start == 0) {
                    frame = !joined.startsWith(bosSpelling);
                    if (frame) out.ids(prefix);
                }
                out.ids(encodeTextRun(joined));
                continue;
            }
            if (start == 0) out.ids(prefix);
            switch (part) {
                case Media.Image image ->
                        media(out, image, tower(towers.projector(Media.Image.class), "image"));
                case Media.Audio audio ->
                        media(out, audio, tower(towers.projector(Media.Audio.class), "audio"));
                case Media.Video video -> {
                    var frames = tower(towers.frames(VIDEO_FRAME_TOKENS), "video");
                    for (Media.Video.Frame image : video.frames())
                        media(out, image.image(), frames);
                }
                case null, default ->
                        throw new IllegalArgumentException(
                                "parts are String or Media.Image, Media.Audio, Media.Video, not "
                                        + part);
            }
        }
        if (parts.isEmpty()) out.ids(prefix);
        if (frame) out.ids(suffix);
        out.finish();
    }

    private <R extends Media> void media(PromptWriter out, R media, MediaProjector<R> projector) {
        boolean audio = media instanceof Media.Audio;
        out.id(SpecialTokens.require(tokenizer, audio ? "<|audio>" : "<|image>"))
                .media(media, null, projector, true)
                .id(SpecialTokens.require(tokenizer, audio ? "<audio|>" : "<image|>"));
    }

    private static <R extends Media> MediaProjector<R> tower(
            Optional<MediaProjector<R>> projector, String medium) {
        return projector.orElseThrow(
                () ->
                        new UnsupportedOperationException(
                                "no tower for "
                                        + medium
                                        + " input attached: load the mmproj with withMedia"));
    }
}
