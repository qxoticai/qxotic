package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;

/**
 * The per-call budget seam of the vision towers: video frames ride the same image embedder at the
 * video processor's own (smaller) soft-token budget, so a tower accepts a resolved budget per
 * encode. The generic {@link com.qxotic.jinfer.Embedder} path keeps the image budget.
 */
interface VisionBudget {
    FloatTensor encode(Media.Image image, int budgetTokens);
}
