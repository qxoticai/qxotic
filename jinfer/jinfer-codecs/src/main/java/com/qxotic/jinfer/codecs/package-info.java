/**
 * Local media decoding into jinfer's canonical model inputs.
 *
 * <p>{@link com.qxotic.jinfer.codecs.ImageCodec} produces RGB images and {@link
 * com.qxotic.jinfer.codecs.AudioCodec} produces 16 kHz mono PCM. On a JVM they use the platform
 * image and sound APIs; native images fall back to ffmpeg. Either backend can be selected
 * explicitly with {@code jinfer.imageDecoder} or {@code jinfer.audioDecoder}. {@link
 * com.qxotic.jinfer.codecs.VideoCodec} decodes sampled video frames through ffmpeg.
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
