package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Objects;

/**
 * A spoken reply's exact record: every position it took, in order - text tokens as their ids, each
 * frame as {@code placeholder} - and the frames' codes. A later turn replays it, so the model reads
 * back what it said (the transcript alone would put text where it was trained on speech) and the
 * prompt cache recognizes the turn it already holds.
 *
 * @param positions the reply's positions, in order
 * @param placeholder the id standing for a frame in {@code positions}
 * @param frames one code array per placeholder, in order
 */
public record SpokenTurn(IntSequence positions, int placeholder, List<int[]> frames) {
    public SpokenTurn {
        Objects.requireNonNull(positions, "positions");
        frames = List.copyOf(frames);
        int placeholders = 0;
        for (int i = 0; i < positions.length(); i++)
            if (positions.intAt(i) == placeholder) placeholders++;
        if (placeholders != frames.size())
            throw new IllegalArgumentException(
                    placeholders + " frame placeholders for " + frames.size() + " frames");
    }
}
