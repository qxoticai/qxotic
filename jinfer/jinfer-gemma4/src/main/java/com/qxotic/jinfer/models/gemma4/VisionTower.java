package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jota.memory.MemoryView;
import java.util.function.Consumer;

/**
 * A Gemma 4 vision tower, which sizes each image to a token budget: still images at {@link
 * VisionPreprocess#IMAGE_TOKEN_BUDGET}, video frames at the budget of the model's video processor
 * ({@link #atBudget}).
 */
interface VisionTower extends MediaProjector<Media.Image> {

    int positions(Media.Image image, int budget);

    void project(Media.Image image, int budget, int maxChunkSize, Consumer<MemoryView<?>> sink);

    /** The tower's geometry; the budget is appended, since it changes every projection. */
    String geometry();

    @Override
    default int positions(Media.Image image) {
        return positions(image, VisionPreprocess.IMAGE_TOKEN_BUDGET);
    }

    @Override
    default void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
        project(image, VisionPreprocess.IMAGE_TOKEN_BUDGET, maxChunkSize, sink);
    }

    @Override
    default String planId() {
        return geometry() + " budget=" + VisionPreprocess.IMAGE_TOKEN_BUDGET;
    }

    /** This tower sizing every image at {@code budget} tokens. */
    default MediaProjector<Media.Image> atBudget(int budget) {
        VisionTower tower = this;
        return new MediaProjector<>() {
            @Override
            public int positions(Media.Image image) {
                return tower.positions(image, budget);
            }

            @Override
            public void project(Media.Image image, int maxChunkSize, Consumer<MemoryView<?>> sink) {
                tower.project(image, budget, maxChunkSize, sink);
            }

            @Override
            public String planId() {
                return tower.geometry() + " budget=" + budget;
            }
        };
    }
}
