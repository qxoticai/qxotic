// Shared image preprocessing for the Gemma 4 vision encoders (gemma4v full ViT and gemma4uv minimal
// embedder): aspect-preserving smart-resize, bilinear HWC->CHW resample, and patch im2col. Pure geometry -
// no weights, no reductions - so the output is bit-identical regardless of who calls it.
package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.Parallel;

final class VisionPreprocess {
    private VisionPreprocess() {}

    /** Default ON. Aspect-preserving like llama.cpp (fewer patches -> faster). Off (-Dvis.squareResize) =
     *  fixed square, for gemma4.java-reference parity. */
    static final boolean SMART_RESIZE = !Boolean.getBoolean("vis.squareResize");

    /** Gemma image token budget: {@code -Djinfer.gemma4.imageTokenBudget=70|140|280|560|1120} — the
     *  preprocess downscale target that sets how many soft tokens an image becomes (280 -> ~256).
     *  Unset -> -1, and each encoder keeps its own default (280, or the GGUF's image_max_pixels). */
    static final int IMAGE_TOKEN_BUDGET = validatedBudget();

    private static int validatedBudget() {
        String p = System.getProperty("jinfer.gemma4.imageTokenBudget");
        if (p == null) return -1;
        int b = Integer.parseInt(p.trim());
        if (b != 70 && b != 140 && b != 280 && b != 560 && b != 1120) {
            throw new IllegalArgumentException("jinfer.gemma4.imageTokenBudget must be 70|140|280|560|1120, got " + b);
        }
        return b;
    }

    /** The effective budget: the property override if set, else {@code deflt}. */
    static int budget(int deflt) { return IMAGE_TOKEN_BUDGET > 0 ? IMAGE_TOKEN_BUDGET : deflt; }

    /** llama.cpp / Qwen2-VL smart_resize: snap each dim to a multiple of {@code factor}=patch·merge,
     *  preserving aspect ratio, bounded by [minPixels, maxPixels]. (640×480, factor 48 -> 624×480.) */
    static int[] smartResize(int w, int h, int factor, int minPixels, int maxPixels) {
        int wb = Math.max(factor, Math.round((float) w / factor) * factor);
        int hb = Math.max(factor, Math.round((float) h / factor) * factor);
        long area = (long) wb * hb;
        if (area > maxPixels) {
            double beta = Math.sqrt((double) w * h / maxPixels);
            wb = Math.max(factor, (int) (Math.floor(w / beta / factor) * factor));
            hb = Math.max(factor, (int) (Math.floor(h / beta / factor) * factor));
        } else if (area < minPixels) {
            double beta = Math.sqrt((double) minPixels / ((double) w * h));
            wb = (int) (Math.ceil(w * beta / factor) * factor);
            hb = (int) (Math.ceil(h * beta / factor) * factor);
        }
        return new int[]{wb, hb};
    }

    /** Bilinear (half-pixel centers) resample of an HWC image to a [3, th, tw] CHW plane, values in [0,1]. */
    static float[] toCHW(Media.Image im, int tw, int th) {
        int plane = th * tw;
        float[] out = new float[3 * plane];
        int H = im.height(), W = im.width(), C = im.channels();
        float[] v = im.values();   // HWC interleaved
        for (int y = 0; y < th; y++) for (int x = 0; x < tw; x++) {
            float fy = (y + 0.5f) * H / th - 0.5f, fx = (x + 0.5f) * W / tw - 0.5f;
            int y0 = Math.max(0, Math.min(H - 1, (int) Math.floor(fy))), y1 = Math.min(H - 1, y0 + 1);
            int x0 = Math.max(0, Math.min(W - 1, (int) Math.floor(fx))), x1 = Math.min(W - 1, x0 + 1);
            float wy = Math.max(0f, fy - y0), wx = Math.max(0f, fx - x0);
            for (int c = 0; c < 3; c++) {
                int cc = Math.min(c, C - 1);
                float a = v[(y0 * W + x0) * C + cc], b = v[(y0 * W + x1) * C + cc],
                      d0 = v[(y1 * W + x0) * C + cc], d1 = v[(y1 * W + x1) * C + cc];
                out[c * plane + y * tw + x] = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + d0 * (1 - wx) * wy + d1 * wx * wy;
            }
        }
        return out;
    }

    /** Patch im2col: each {@code ps×ps} patch in channel-outer [c, ky, kx] order, pixels scaled to [-1, 1],
     *  into a [nPatches, 3·ps·ps] row-major tensor (nPatches = (tw/ps)·(th/ps)). */
    static FloatTensor im2col(Media.Image image, int tw, int th, int ps) {
        int px = tw / ps, py = th / ps, n = px * py, patchVec = 3 * ps * ps, plane = th * tw;
        float[] chw = toCHW(image, tw, th);
        FloatTensor flat = FloatTensor.allocateF32(n * patchVec);
        Parallel.forRows(n, gi -> {
            int gy = gi / px, gx = gi % px, row = gi * patchVec, w = 0;
            for (int c = 0; c < 3; c++) for (int ky = 0; ky < ps; ky++) for (int kx = 0; kx < ps; kx++) {
                float pix = chw[c * plane + (gy * ps + ky) * tw + (gx * ps + kx)] * 2f - 1f;
                flat.setFloat((long) row + w++, pix);
            }
        });
        return flat;
    }
}
