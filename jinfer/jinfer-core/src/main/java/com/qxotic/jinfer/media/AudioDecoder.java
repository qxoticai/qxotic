// Pluggable audio decoder: encoded audio bytes (mp3/wav/flac/ogg/m4a/...) -> 16 kHz MONO float PCM
// (Media.Audio, float samples, 1 channel, 16000 Hz - the universal speech-encoder input). Two impls:
//   - FfmpegAudioDecoder: shells out to ffmpeg (decode + resample + downmix); native-image-safe,
//     cross-platform, broad format support; needs ffmpeg on PATH.
//   - JavaSoundAudioDecoder: javax.sound.sampled for WAV/AIFF/AU (no external process), ffmpeg fallback for
//     compressed formats; JVM-only (AudioSystem's ServiceLoader/SPI is fragile under GraalVM native-image).
// Unlike the image codec (decode only), the audio decoder ALSO resamples to 16 kHz + downmixes to mono,
// because that target is universal across speech encoders (not a per-model choice like the image resize).
// Select via AudioCodec (-Djinfer.audioDecoder=ffmpeg|javasound; default ffmpeg under native-image, javasound on JVM).
package com.qxotic.jinfer.media;

import com.qxotic.jinfer.Media;

import java.io.IOException;
import java.nio.file.Path;

public interface AudioDecoder {

    /** Decode an audio file into 16 kHz mono float PCM. */
    Media.Audio load(Path path) throws IOException;

    /** Decode encoded audio bytes into 16 kHz mono float PCM. */
    Media.Audio decode(byte[] encoded) throws IOException;

    /** Short backend name, for logging/diagnostics ("ffmpeg" / "javasound"). */
    String name();
}
