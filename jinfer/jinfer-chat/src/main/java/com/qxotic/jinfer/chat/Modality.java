package com.qxotic.jinfer.chat;

/**
 * What a reply may be made of, allowed on {@link ChatEngine.Request#output}. The reply holds only
 * allowed outputs, and where several are allowed the model chooses: text and audio allowed, a reply
 * in text alone is a fine answer. A model never produces what it lacks, and refuses a request when
 * it can produce none of the allowed outputs - audio alone from a model that cannot speak. Text
 * arrives as {@link ChatEngine.Delta}s and {@link Content.Text}; generated media as {@link
 * ChatEngine.ReplySink#onMedia} parts, speech also whole as the reply's {@link Content.Media}.
 */
public enum Modality {
    /** Words: the transcript, when the reply also speaks. */
    TEXT,
    /** Speech in the model's own voice. */
    AUDIO,
    /** Generated images; no port draws yet. */
    IMAGE,
    /** Generated video; no port films yet. */
    VIDEO
}
