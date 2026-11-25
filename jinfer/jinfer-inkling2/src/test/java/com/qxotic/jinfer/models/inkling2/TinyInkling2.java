package com.qxotic.jinfer.models.inkling2;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * A small random {@code inkling2_loop} checkpoint, the same bytes on every run: eight layers in
 * blocks of two, the middle four looped three times, a window and a bias extent short enough for a
 * test prompt to pass both. Its vocabulary is the 256 bytes, fillers and the chat control tokens.
 */
final class TinyInkling2 {
    static final int DIM = 64, HEADS = 4, KV_HEADS = 2, HEAD_SIZE = 16;
    static final int LAYERS = 8, BLOCK = 2, LOOP_START = 2, LOOP_END = 6, LOOP_COUNT = 3, RANK = 8;
    static final int RELATIVE = 4, EXTENT = 12, WINDOW = 16, CONV = 4;
    static final int EXPERTS = 8, USED = 2, FF = 32, VOCABULARY = 300, PADDED = 320;

    /** The vision tower: a position table of 4 x 4 for grids of other sizes to resize. */
    static final int PATCH = 4, MERGE = 2, VISION_DIM = 32, VISION_HEADS = 2, VISION_FF = 48;

    static final int VISION_LAYERS = 2, POSITION_SIDE = 4, PROJECTOR = 40;

    private static final String[] SPECIALS = {
        "<|endoftext|>",
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_text|>",
        "<|content_image|>",
        "<|content_model_end_sampling|>",
        "<|content_thinking|>",
        "<|end_message|>",
        "<|content_xml|>",
        "<|content_invoke_tool_json|>"
    };

    private final Builder gguf = Builder.newBuilder();
    private final Map<String, float[]> data = new LinkedHashMap<>();
    private final Random random = new Random(7);

    /** Writes the checkpoint to {@code path}. */
    static Path write(Path path) throws IOException {
        TinyInkling2 tiny = new TinyInkling2();
        tiny.metadata();
        tiny.tensors();
        return tiny.save(path);
    }

    /** Writes the checkpoint's mmproj, an {@code inkling2v} vision tower, to {@code path}. */
    static Path writeVision(Path path) throws IOException {
        TinyInkling2 tiny = new TinyInkling2();
        tiny.vision();
        return tiny.save(path);
    }

    private Path save(Path path) throws IOException {
        GGUF header = gguf.build();
        try (FileChannel channel =
                FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            GGUF.write(header, channel);
            for (Map.Entry<String, float[]> tensor : data.entrySet()) {
                ByteBuffer bytes =
                        ByteBuffer.allocate(tensor.getValue().length * Float.BYTES)
                                .order(ByteOrder.LITTLE_ENDIAN);
                bytes.asFloatBuffer().put(tensor.getValue());
                channel.write(bytes, header.absoluteOffset(header.getTensor(tensor.getKey())));
            }
        }
        return path;
    }

    private void metadata() {
        String a = "inkling2_loop.";
        boolean[] swa = new boolean[LAYERS];
        int[] heads = new int[LAYERS], kvHeads = new int[LAYERS];
        for (int layer = 0; layer < LAYERS; layer++) {
            swa[layer] = layer % BLOCK != BLOCK - 1; // a block's last layer attends to everything
            heads[layer] = HEADS;
            kvHeads[layer] = KV_HEADS;
        }
        String[] tokens = new String[VOCABULARY];
        int[] types = new int[VOCABULARY];
        for (int id = 0; id < VOCABULARY; id++) {
            int special = id - (VOCABULARY - SPECIALS.length);
            tokens[id] =
                    id < 256
                            ? byteToken(id)
                            : special < 0 ? "<|filler_" + id + "|>" : SPECIALS[special];
            types[id] = id < 256 ? 1 : 3;
        }
        gguf.putString("general.architecture", "inkling2_loop")
                .putUnsignedInteger(a + "block_count", LAYERS)
                .putUnsignedInteger(a + "context_length", 512)
                .putUnsignedInteger(a + "embedding_length", DIM)
                .putUnsignedInteger(a + "feed_forward_length", FF)
                .putArrayOfUnsignedInteger(a + "attention.head_count", heads)
                .putArrayOfUnsignedInteger(a + "attention.head_count_kv", kvHeads)
                .putFloat(a + "attention.layer_norm_rms_epsilon", 1e-6f)
                .putUnsignedInteger(a + "attention.key_length", HEAD_SIZE)
                .putUnsignedInteger(a + "attention.value_length", HEAD_SIZE)
                .putUnsignedInteger(a + "attention.key_length_swa", HEAD_SIZE)
                .putUnsignedInteger(a + "attention.value_length_swa", HEAD_SIZE)
                .putUnsignedInteger(a + "rope.dimension_count", 0)
                .putUnsignedInteger(a + "vocab_size", PADDED)
                .putUnsignedInteger(a + "vocab_size_unpadded", VOCABULARY)
                .putUnsignedInteger(a + "expert_count", EXPERTS)
                .putUnsignedInteger(a + "expert_used_count", USED)
                .putUnsignedInteger(a + "expert_feed_forward_length", FF)
                .putFloat(a + "expert_weights_scale", 2f)
                .putFloat(a + "activation.situ_beta", 4f)
                .putFloat(a + "activation.situ_linear_beta", 25f)
                .putUnsignedInteger(a + "attention.sliding_window", WINDOW)
                .putArrayOfBoolean(a + "attention.sliding_window_pattern", swa)
                .putUnsignedInteger(a + "shortconv.l_cache", CONV)
                .putUnsignedInteger(a + "attn_res.block_size", BLOCK)
                .putFloat(a + "final_logit_softcapping", 30f)
                .putUnsignedInteger(a + "attention.relative_length", RELATIVE)
                .putUnsignedInteger(a + "attention.relative_extent", EXTENT)
                .putFloat(a + "attention.qk_clip", 5f)
                .putFloat(a + "attention.relative_clip", 3f)
                .putUnsignedInteger(a + "attention.log_scale_floor", 6)
                .putFloat(a + "attention.log_scale_alpha", 0.1f)
                .putUnsignedInteger(a + "loop.start", LOOP_START)
                .putUnsignedInteger(a + "loop.end", LOOP_END)
                .putUnsignedInteger(a + "loop.count", LOOP_COUNT)
                .putUnsignedInteger(a + "loop.gate_rank", RANK)
                .putString("tokenizer.ggml.model", "gpt2")
                .putString("tokenizer.ggml.pre", "inkling2")
                .putArrayOfString("tokenizer.ggml.tokens", tokens)
                .putArrayOfInteger("tokenizer.ggml.token_type", types)
                .putArrayOfString("tokenizer.ggml.merges", new String[0])
                .putUnsignedInteger("tokenizer.ggml.eos_token_id", VOCABULARY - 5)
                .putBoolean("tokenizer.ggml.add_bos_token", false);
    }

    /** GPT-2's spelling of a byte: itself when printable, else a code point from 256 up. */
    private static String byteToken(int value) {
        int shifted = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b != 0xAD);
            if (b == value) return String.valueOf((char) (printable ? b : 256 + shifted));
            if (!printable) shifted++;
        }
        throw new IllegalArgumentException("not a byte: " + value);
    }

    private void tensors() {
        int queryDim = HEADS * HEAD_SIZE, kvDim = KV_HEADS * HEAD_SIZE;
        tensor("token_embd.weight", 0f, 1f, DIM, PADDED);
        tensor("token_embd_norm.weight", 1f, .2f, DIM);
        tensor("output.weight", 0f, .5f, DIM, PADDED);
        tensor("output_norm.weight", 1f, .2f, DIM);
        tensor("output_res_query.weight", 0f, 1f, DIM);
        for (int layer = 0; layer < LAYERS; layer++) {
            String p = "blk." + layer + ".";
            boolean swa = layer % BLOCK != BLOCK - 1;
            tensor(p + "attn_q.weight", 0f, .1f, DIM, queryDim);
            tensor(p + "attn_k.weight", 0f, .1f, DIM, kvDim);
            tensor(p + "attn_v.weight", 0f, .1f, DIM, kvDim);
            tensor(p + "attn_output.weight", 0f, .1f, queryDim, DIM);
            tensor(p + "attn_gate.weight", 0f, .1f, DIM, queryDim);
            tensor(p + "attn_q_clip.weight", 1f, .3f, HEAD_SIZE, HEADS);
            tensor(p + "attn_sink_proj.weight", 0f, .1f, DIM, HEADS);
            tensor(p + "attn_rel.weight", 0f, .1f, DIM, RELATIVE * HEADS);
            tensor(p + "attn_rel_bias.weight", 0f, .5f, RELATIVE, swa ? WINDOW : EXTENT);
            tensor(p + "attn_rel_clip.weight", 1f, .3f, RELATIVE, HEADS);
            tensor(p + "attn_k_conv.weight", 0f, .5f, CONV, kvDim);
            tensor(p + "attn_v_conv.weight", 0f, .5f, CONV, kvDim);
            tensor(p + "attn_post_conv.weight", 0f, .5f, CONV, DIM);
            tensor(p + "ffn_post_conv.weight", 0f, .5f, CONV, DIM);
            if (swa) {
                tensor(p + "attn_norm.weight", 1f, .2f, DIM);
            } else {
                for (String input : new String[] {"q", "k", "v"})
                    tensor(p + "attn_" + input + "_inp_norm.weight", 1f, .2f, DIM);
                tensor(p + "attn_k_conv_proj.weight", 0f, .1f, DIM, CONV * KV_HEADS);
                tensor(p + "attn_v_conv_proj.weight", 0f, .1f, DIM, CONV * KV_HEADS);
            }
            if (layer > 0 && swa) tensor(p + "attn_res_query.weight", 0f, 1f, DIM);
            if (layer > 0 && !swa) tensor(p + "attn_res_query.weight", 0f, 1f, DIM, 3);
            tensor(p + "ffn_res_query.weight", 0f, 1f, DIM);
            tensor(p + "ffn_norm.weight", 1f, .2f, DIM);
            tensor(p + "ffn_gate_inp.weight", 0f, .3f, DIM, EXPERTS);
            // a wide selection bias: FP16 rounding in the history must not flip a route
            tensor(p + "exp_probs_b.bias", 0f, 2f, EXPERTS);
            tensor(p + "ffn_gate_up_exps.weight", 0f, .1f, DIM, 2 * FF, EXPERTS);
            tensor(p + "ffn_down_exps.weight", 0f, .1f, FF, DIM, EXPERTS);
            if (layer < LOOP_START || layer >= LOOP_END) continue;
            tensor(p + "loop_msg_query.weight", 0f, 1f, DIM, 4);
            for (String input : new String[] {"q", "k", "v", "ffn"}) {
                tensor(p + "loop_" + input + "_ctrl_down.weight", 0f, .1f, 2 * DIM, RANK);
                tensor(p + "loop_" + input + "_ctrl_gain.weight", 0f, .3f, RANK, DIM);
                tensor(p + "loop_" + input + "_ctrl_forget.weight", 0f, .3f, RANK, DIM);
            }
        }
    }

    private void vision() {
        String v = "clip.vision.";
        gguf.putString("general.architecture", "clip")
                .putString("general.type", "mmproj")
                .putBoolean("clip.has_vision_encoder", true)
                .putString(v + "projector_type", "inkling2v")
                .putUnsignedInteger(v + "projection_dim", DIM)
                .putUnsignedInteger(v + "image_size", PATCH * POSITION_SIDE)
                .putUnsignedInteger(v + "patch_size", PATCH)
                .putUnsignedInteger(v + "spatial_merge_size", MERGE)
                .putUnsignedInteger(v + "embedding_length", VISION_DIM)
                .putUnsignedInteger(v + "feed_forward_length", VISION_FF)
                .putUnsignedInteger(v + "block_count", VISION_LAYERS)
                .putUnsignedInteger(v + "attention.head_count", VISION_HEADS)
                .putFloat(v + "attention.layer_norm_epsilon", 1e-5f)
                .putArrayOfFloat(v + "image_mean", new float[] {.5f, .5f, .5f})
                .putArrayOfFloat(v + "image_std", new float[] {.5f, .5f, .5f});
        tensor("v.class_embd", 0f, 1f, VISION_DIM);
        tensor("v.class_position_embd.weight", 0f, 1f, VISION_DIM);
        tensor("v.patch_embd.weight", 0f, .3f, PATCH, PATCH, 3, VISION_DIM);
        tensor("v.position_embd.weight", 0f, 1f, VISION_DIM, POSITION_SIDE * POSITION_SIDE);
        tensor("v.pre_ln.weight", 1f, .2f, VISION_DIM);
        tensor("v.pre_ln.bias", 0f, .2f, VISION_DIM);
        for (int layer = 0; layer < VISION_LAYERS; layer++) {
            String p = "v.blk." + layer + ".";
            for (String norm : new String[] {"ln1", "ln2"}) {
                tensor(p + norm + ".weight", 1f, .2f, VISION_DIM);
                tensor(p + norm + ".bias", 0f, .2f, VISION_DIM);
            }
            for (String linear : new String[] {"attn_q", "attn_k", "attn_v", "attn_out"}) {
                tensor(p + linear + ".weight", 0f, .3f, VISION_DIM, VISION_DIM);
                tensor(p + linear + ".bias", 0f, .2f, VISION_DIM);
            }
            tensor(p + "ffn_up.weight", 0f, .3f, VISION_DIM, VISION_FF);
            tensor(p + "ffn_up.bias", 0f, .2f, VISION_FF);
            tensor(p + "ffn_down.weight", 0f, .3f, VISION_FF, VISION_DIM);
            tensor(p + "ffn_down.bias", 0f, .2f, VISION_DIM);
            tensor(p + "ls1.weight", .5f, .2f, VISION_DIM);
            tensor(p + "ls2.weight", .5f, .2f, VISION_DIM);
        }
        tensor("mm.1.weight", 0f, .2f, VISION_DIM * MERGE * MERGE, PROJECTOR);
        tensor("mm.2.weight", 0f, .2f, PROJECTOR, DIM);
    }

    /** A gaussian FP32 tensor; {@code shape} is in GGUF order, the innermost axis first. */
    private void tensor(String name, float mean, float deviation, long... shape) {
        int size = 1;
        for (long axis : shape) size *= (int) axis;
        float[] values = new float[size];
        for (int i = 0; i < size; i++) values[i] = mean + deviation * (float) random.nextGaussian();
        data.put(name, values);
        gguf.putTensor(TensorEntry.create(name, shape, GGMLType.F32, 0));
    }
}
