package com.qxotic.jinfer.x.telemetry;

import com.qxotic.jinfer.x.boundary.Media;
import jdk.jfr.Category;
import jdk.jfr.Description;
import jdk.jfr.Event;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;
import jdk.jfr.Threshold;
import jdk.jfr.Timespan;

/** One actual media-to-prompt projection. Cache hits emit no event. */
@Name("jinfer.MediaProjection")
@Label("Media Projection")
@Category({"jinfer", "Inference"})
@Description("One actual media-to-prompt projection; cache hits emit no event.")
@StackTrace(false)
@Threshold("0 ms")
public final class MediaProjectionEvent extends Event {

    /** Starts an event carrying bounded modality and decoded-source size information. */
    public static MediaProjectionEvent started(Media source) {
        MediaProjectionEvent event = new MediaProjectionEvent();
        event.errorType = "";
        switch (source) {
            case Media.Image image -> event.image(image);
            case Media.Audio audio -> {
                event.modality = "audio";
                event.sourceChannels = audio.channels();
                event.sourceSampleRate = audio.sampleRate();
                long frames = audio.pcm().length / audio.channels();
                event.sourceDuration = frames * 1_000_000_000L / audio.sampleRate();
            }
            case Media.Video video -> {
                event.modality = "video";
                event.sampledFrames = video.frames().size();
                if (!video.frames().isEmpty()) {
                    Media.Image first = video.frames().getFirst().image();
                    boolean uniform =
                            video.frames().stream()
                                    .map(Media.Video.Frame::image)
                                    .allMatch(
                                            image ->
                                                    image.width() == first.width()
                                                            && image.height() == first.height()
                                                            && image.channels()
                                                                    == first.channels());
                    if (uniform) event.imageShape(first);
                }
            }
        }
        event.begin();
        return event;
    }

    private void image(Media.Image image) {
        modality = "image";
        imageShape(image);
    }

    private void imageShape(Media.Image image) {
        sourceWidth = image.width();
        sourceHeight = image.height();
        sourceChannels = image.channels();
    }

    @Label("Modality")
    public String modality;

    @Label("Source Width")
    public int sourceWidth;

    @Label("Source Height")
    public int sourceHeight;

    @Label("Source Channels")
    public int sourceChannels;

    @Label("Sampled Frames")
    public int sampledFrames;

    @Label("Source Sample Rate")
    public int sourceSampleRate;

    @Label("Source Duration")
    @Timespan(Timespan.NANOSECONDS)
    public long sourceDuration;

    /** Empty on success; a class name on failure, never an unbounded message. */
    @Label("Error Type")
    public String errorType;
}
