package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.nio.file.Path;

/**
 * A video sampling policy: a source file to timestamped frames. Any policy is a timestamp list
 * ({@link VideoCodec#framesAt}), so a sampler is just the function that picks the list - the named
 * policies compose as lambdas: {@code v -> VideoCodec.ffmpeg().uniform(v, 16)}, {@code v ->
 * VideoCodec.ffmpeg().span(v, 8)}, a window of a long source, scene cuts, caller-curated moments.
 * Whatever the policy, frames carry their TRUE timestamps, which is what grounds the model.
 */
@FunctionalInterface
public interface VideoSampler {

    /**
     * The reference default: {@link VideoCodec#DEFAULT_NUM_FRAMES} frames uniform across the whole
     * duration.
     */
    VideoSampler UNIFORM = v -> VideoCodec.ffmpeg().uniform(v);

    Media.Video sample(Path video) throws IOException;
}
