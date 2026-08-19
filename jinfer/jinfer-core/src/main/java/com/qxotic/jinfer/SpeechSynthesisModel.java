package com.qxotic.jinfer;

import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.function.Predicate;

/** A text-to-waveform model with reusable runtime state. */
public interface SpeechSynthesisModel<C, W, S extends RuntimeState> extends Model<C, W, S> {

    /** Creates state that owns its memory. */
    S newState();

    /** Creates state that borrows caller-owned memory. */
    S newState(MemoryArena<MemorySegment> arena);

    /** Streams waveform clips; the sink returns false to cancel after the current clip. */
    void speak(S state, String text, SpeechOptions options, Predicate<Media.Audio> sink);

    /** Synthesizes and concatenates the complete waveform. */
    default Media.Audio speak(S state, String text, SpeechOptions options) {
        var clips = new ArrayList<Media.Audio>();
        speak(state, text, options, clips::add);
        return Media.Audio.concat(clips);
    }
}
