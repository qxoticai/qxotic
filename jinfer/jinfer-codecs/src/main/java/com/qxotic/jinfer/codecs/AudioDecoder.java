package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Encoded audio bytes to 16 kHz mono float PCM, which is what every speech encoder expects. Two
 * implementations, selected by {@link AudioCodec}: ffmpeg, and {@code javax.sound} for WAV/AIFF/AU
 * with no external process.
 */
public interface AudioDecoder {

    /** Decode an audio file into 16 kHz mono float PCM. */
    Media.Audio load(Path path) throws IOException;

    /** Decode encoded audio bytes into 16 kHz mono float PCM. */
    Media.Audio decode(byte[] encoded) throws IOException;

    /**
     * Backend id, as {@code ImageDecoder.name()}: part of a media cache's identity, because two
     * decoders may turn the same encoded bytes into different samples.
     */
    String name();
}
