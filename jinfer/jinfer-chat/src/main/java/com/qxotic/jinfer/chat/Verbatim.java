package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.toknroll.IntSequence;
import java.util.Objects;

/**
 * How a generated turn was said: every position it took, from the reply prefix the prompt owned to
 * its last token, and the model that said it. A later prompt writes the record in place of the turn
 * ({@link PromptWriter#replay}), so the model reads back exactly what it wrote - re-encoding the
 * text would tokenize differently, and speech has no text form - and the prompt cache recognizes
 * the stream it already holds. A speech frame is {@link #FRAME} in {@code positions}; its codes
 * follow in {@code codes}, frame after frame, the same number each.
 *
 * <p>Carried by the {@link Message} the engine returns; {@link #NONE} on a turn nobody generated.
 * The engine replays a record only into the model that wrote it: a turn with another model's record
 * renders from its parts, like any other.
 *
 * @param model the seed key of the model that generated the turn, companions included
 * @param positions the turn's token ids in order, a speech frame as {@link #FRAME}
 * @param codes every frame's codes, in frame order
 */
public record Verbatim(ContentKey model, IntSequence positions, IntSequence codes) {

    /** The position a speech frame took; its codes are the next {@link #codebooks()} of them. */
    public static final int FRAME = -1;

    /** The record of a turn nobody generated: nothing to replay. */
    public static final Verbatim NONE =
            new Verbatim(new ContentKey("none"), IntSequence.empty(), IntSequence.empty());

    public Verbatim {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(positions, "positions");
        Objects.requireNonNull(codes, "codes");
        int frames = frames(positions);
        if (frames == 0 ? codes.length() != 0 : codes.length() % frames != 0 || codes.length() == 0)
            throw new IllegalArgumentException(
                    codes.length() + " codes for " + frames + " speech frames");
    }

    /** Whether there is nothing to replay: {@link #NONE}, or a turn that took no position. */
    public boolean isEmpty() {
        return positions.isEmpty();
    }

    /** How many speech frames the turn took. */
    public int frames() {
        return frames(positions);
    }

    /** Codes per speech frame; zero when the turn took none. */
    public int codebooks() {
        int frames = frames();
        return frames == 0 ? 0 : codes.length() / frames;
    }

    private static int frames(IntSequence positions) {
        int frames = 0;
        for (int i = 0; i < positions.length(); i++) if (positions.intAt(i) == FRAME) frames++;
        return frames;
    }
}
