package com.qxotic.jinfer.server;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.Multimodal;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * The media content parts every endpoint takes, in OpenAI's chat syntax: {@code image_url} (or
 * {@code input_image}) as a base64 data URI, {@code input_audio} as base64 {@code data}, {@code
 * video_url} as a base64 data URI. The server never fetches a remote URL.
 */
final class MediaParts {
    private MediaParts() {}

    /** A part's modality - text, image, audio or video - or null for a type no endpoint takes. */
    static String modality(Map<String, Object> part) {
        return switch (Values.stringValue(part.get("type"), "")) {
            case "text", "input_text" -> "text";
            case "image_url", "input_image" -> "image";
            case "input_audio" -> "audio";
            case "video_url" -> "video";
            default -> null;
        };
    }

    static String text(Map<String, Object> part) {
        return Values.stringValue(
                part.get("text") != null ? part.get("text") : part.get("input_text"), "");
    }

    /**
     * Text, and what the model's projectors take, read off without instantiating anything; video
     * rides the vision projector, so it is listed only where the endpoint renders it.
     */
    static List<String> modalities(Object model, boolean video) {
        List<String> modalities = new ArrayList<>(List.of("text"));
        if (model instanceof Multimodal media) {
            if (media.projector(Media.Image.class).isPresent()) {
                modalities.add("image");
                if (video) modalities.add("video");
            }
            if (media.projector(Media.Audio.class).isPresent()) modalities.add("audio");
        }
        return List.copyOf(modalities);
    }

    static Content.Media image(Map<String, Object> part) {
        Object image = part.get("image_url") != null ? part.get("image_url") : part.get("image");
        byte[] bytes = dataUri(url(image), "image_url");
        try {
            return new Content.Media(ImageCodec.decode(bytes), ContentKey.sha256(bytes));
        } catch (IOException failure) {
            throw new IllegalArgumentException(
                    "image could not be decoded: " + failure.getMessage());
        }
    }

    static Content.Media audio(Map<String, Object> part) {
        Map<String, Object> audio = Values.asObject(part.get("input_audio"), "input_audio");
        byte[] bytes;
        try {
            bytes = Base64.getDecoder().decode(Values.stringValue(audio.get("data"), ""));
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("input_audio data is not valid base64");
        }
        try {
            return new Content.Media(AudioCodec.decode(bytes), ContentKey.sha256(bytes));
        } catch (IOException failure) {
            throw new IllegalArgumentException(
                    "audio could not be decoded: " + failure.getMessage());
        }
    }

    /**
     * A {@code video_url} data URI, spilled to a file for ffmpeg and sampled by {@code sampler}.
     */
    static Content.Media video(Map<String, Object> part, VideoSampler sampler) {
        MediaSpill.Spilled spilled = null;
        try {
            spilled = MediaSpill.base64Video(url(part.get("video_url")), "video_url");
            Media.Video video = sampler.sample(spilled.file());
            return new Content.Media(video, spilled.key());
        } catch (IOException failure) {
            throw new IllegalArgumentException(
                    "video could not be decoded: " + failure.getMessage());
        } finally {
            if (spilled != null) {
                MediaSpill.deleteQuietly(spilled.file());
            }
        }
    }

    private static String url(Object value) {
        return value instanceof Map<?, ?> map
                ? Values.stringValue(map.get("url"), "")
                : Values.stringValue(value, "");
    }

    private static byte[] dataUri(String value, String field) {
        Validation.require(
                value.startsWith("data:"),
                "%s must be a data: URI (the server does not fetch remote URLs)",
                field);
        int comma = value.indexOf(',');
        Validation.require(
                comma > 0 && value.substring(0, comma).endsWith(";base64"),
                "%s data: URI must be base64-encoded",
                field);
        try {
            return Base64.getDecoder().decode(value.substring(comma + 1));
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException(field + " base64 payload is malformed");
        }
    }
}
