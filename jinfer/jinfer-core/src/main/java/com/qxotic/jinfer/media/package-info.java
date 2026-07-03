/**
 * Media codecs: file/stream bytes to the canonical decoded signals in
 * {@link com.qxotic.jinfer.Media} (RGB raster, PCM waveform) - the step BEFORE the model.
 *
 * <p>Each modality has a facade ({@link com.qxotic.jinfer.media.ImageCodec},
 * {@link com.qxotic.jinfer.media.AudioCodec}) over pluggable decoders: Java platform APIs on the
 * JVM, ffmpeg on native images (where {@code javax.imageio}/{@code javax.sound} are unavailable;
 * the Java backends load reflectively so they stay out of the closed world), overridable via
 * {@code -Djinfer.imageDecoder}/{@code -Djinfer.audioDecoder}. Downstream of here, media is the
 * model's business: {@code MultiModal.embedder} turns the decoded signal into model-dim rows.
 */
package com.qxotic.jinfer.media;
