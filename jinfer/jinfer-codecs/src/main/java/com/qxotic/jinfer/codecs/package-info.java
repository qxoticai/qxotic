/**
 * Local media decoding into jinfer's canonical model inputs.
 *
 * <p>{@link com.qxotic.jinfer.codecs.ImageCodec} produces RGB images and {@link
 * com.qxotic.jinfer.codecs.AudioCodec} produces 16 kHz mono PCM, through ffmpeg wherever it is on
 * PATH. A JVM without it uses the platform image and sound APIs, which read PNG, JPEG, GIF, BMP and
 * TIFF, and WAV, AIFF and AU. Either backend can be selected explicitly with {@code
 * jinfer.imageDecoder} or {@code jinfer.audioDecoder}, and a selected backend is the only one
 * asked. {@link com.qxotic.jinfer.codecs.VideoCodec} decodes sampled video frames through ffmpeg.
 * {@link com.qxotic.jinfer.codecs.Espeak} is the other external tool: espeak-ng as
 * grapheme-to-phoneme, probed on demand, over which the speech ports build their phonemizers.
 *
 * <p>Decoding is separate from model projection: codecs know files and bytes, while a model's
 * {@link com.qxotic.jinfer.media.MediaProjector} turns decoded media into borrowed embedding rows.
 * No codec fetches remote content.
 *
 * <p><b>Internal to jinfer: published so the dependency graph resolves, with no API or
 * compatibility promise. Depend on jinfer-core, the model artifacts, or the framework adapters
 * instead.</b>
 */
package com.qxotic.jinfer.codecs;
