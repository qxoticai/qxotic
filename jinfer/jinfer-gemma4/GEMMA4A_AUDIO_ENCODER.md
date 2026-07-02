# Gemma 4 E2B audio encoder (projector_type `gemma4a`) - port spec

Reverse-engineered from the reference `../llama.cpp` (a future/fictional fork) across
`tools/mtmd/{clip.cpp,clip-impl.h,clip-model.h,mtmd-audio.cpp,models/gemma4a.cpp}`.
The E2B (`gemma-4-E2B-it-GGUF/mmproj-F32.gguf`) uses `PROJECTOR_TYPE_GEMMA4A` = a real
24-block **Conformer** ASR encoder. This is NOT the 12B's encoder-free `gemma4ua`
(single `mm.a.input_projection` matmul, already ported in `Gemma4Audio.java`).

## Status
- **Mel front-end**: implemented here (`AudioPreprocess.java`) - self-contained, spec below.
- **Conformer body**: NOT implemented. Fully specified below. It needs ~5 primitives jinfer
  lacks and cannot be brought to parity without iterative debugging, so it is out of scope
  for a single pass. Recommend implementing scalar-correct first (parity), then batching.

## Mel front-end (`clip.cpp` GEMMA4A hparams + `mtmd-audio.cpp`)
- sample_rate 16000, `n_fft = 512` (bins = 257), Hann `window_len = 320` (20 ms, periodic),
  zero-padded to 512, `hop = 160` (10 ms). `eps = 1e-6`.
- Per frame: Hann-window 320 samples -> zero-pad to 512 -> real FFT -> **power** spectrum
  `re^2 + im^2` (not magnitude) -> mel filterbank matmul -> `max(sum, mel_floor)` -> log.
- **Mel filterbank** (`fill_mel_filterbank_matrix`, defaults `slaney_area_norm=true`,
  `use_htk=false`): the NON-HTK custom scale - `min_log_hz=1000`, `lin_slope`,
  `log_step = log(6.4)/27`, `min_log_mel = 1000*lin_slope`; Slaney area normalization
  `enorm = 2/(f_right-f_left)`. `n_mel = clip.audio.num_mel_bins`.
- Confirm from the gemma preprocess params: `use_magnitude` (expect false=power),
  `use_natural_log` (log vs log10), `mel_floor`, and the boundary padding of `samples_padded`.
- Output: `[n_mel, n_frames]`, fed as `build_inp_raw(1)` then transposed.

## Conformer body (`models/gemma4a.cpp::build()`) - full op order
Constants: `res_weight=0.5`, `norm_eps=1e-6`, all norms are **RMS** (except the two conv2d
LayerNorms). Hyperparams from `clip.audio.*`: `block_count=24`, `attention.head_count`,
`embedding_length`, `feed_forward_length`, `projection_dim`, `num_mel_bins`.

**1. Subsampling (4x downsample):** for i in {0,1}: `conv2d(sscp_conv_w[i], stride=2,2 pad=1,1)`
+ bias -> LayerNorm over channels (permute, norm, *weight, permute back) -> **ReLU**. Then
flatten `[freq,time,ch] -> [ch*freq, time]` -> `input_projection` (+bias). Input is the mel,
transposed.

**2. Chunked LOCAL attention params (NOT full attention):** chunk `C=12`, past horizon
`P=12`, context `S=C+P=24`, RPE positions `R=P+1=13`, blocks `B=ceil(n_pos/C)`, padded
`Np=B*C`. Two extra inputs: `pos_emb [n_head*d_head, R]` (sinusoidal RPE) and blocked
`kq_mask [S, C, B]`.

**3. Each of 24 blocks (residual threaded through as `residual`):**
- **FFN1 (half-step)**: RMS(ff_norm) -> FFN_SILU(ff_up, ff_down) -> RMS(ff_post_norm) ->
  `residual += 0.5 * ffn`.
- **Attention (chunked local + Transformer-XL RPE + softcap)**:
  - `q_scale = (1/sqrt(d_head))/ln2`, `k_scale = ln(1+e)/ln2` (softplus(1)/ln2), `softcap=50`.
  - RMS(attn_pre_norm) on residual -> Q,K,V = build_mm (clamped).
  - reshape `[d_head, n_head, n_pos]`. `Q *= q_scale`, then `Q *= per_dim_scale_w` (per d_head).
    `K *= k_scale`, then `K *= per_dim_k_scale_w`.
  - **Q blocking**: pad to Np, reshape `[D,H,C,B]`, permute -> `[D,C,B,H]`.
  - **K/V overlapping-window blocking** (the hard part): pad to `S*B`, `ggml_roll` right by P
    (left-pad), materialize, then a strided `view_4d` with block stride `C` (overlap since C<S)
    to get `[D,H,S,B]`; permute K->`[D,S,B,H]`, V->`[S,D,B,H]`.
  - `matrix_ac = Kblk^T @ Qcur -> [S,C,B,H]`.
  - **RPE**: `attn_k_rel_w @ pos_emb -> [D,H,R] -> [D,R,H]`; `Q_flat @ RPE -> [R,C*B,H] ->
    [R,C,B,H]`; **blocked relative shift** (Transformer-XL appendix B: pad `S+1-R` on dim0,
    reshape `(S+1)*C`, strided view `C*S`, reshape `[S,C,B,H]`); `matrix_ac += matrix_bd`.
  - **Softcap**: `*1/50 -> tanh -> *50`. Add `kq_mask`. `softmax`.
  - `attn @ Vblk -> [D,C,B,H]` -> permute `[D,H,C,B]` -> flatten -> trim pad to `n_pos`.
  - `o_w` (+o_b, clamped), optional RMS(attn_post_norm). `residual += x`.
- **Conv module**: RMS(norm_conv) -> `conv_pw1` (clamped) -> **GLU** (split halves,
  `x = first * sigmoid(second)`, transpose) -> **causal depthwise conv1d** (`ggml_ssm_conv`
  after pad4+roll4 for left-only pad) with `conv_dw_w` (+conv_dw_b) -> RMS(conv_norm) ->
  **SiLU** -> `conv_pw2` (clamped). `residual += x`.
- **FFN2 (half-step)**: like FFN1 with `ff_*_1` tensors. `residual += 0.5 * ffn`.
- **Layer out**: optional RMS(ln_2).

**4. Output**: `audio_out_proj` (+bias, `a.pre_encode.out`) -> `rms_norm` -> `* mm_soft_emb_norm`
-> `mm_input_projection` (-> text `embedding_length`, 3840).

**5. ClippableLinear (`build_mm`)**: for any weight with clamp scalars, `clamp(x, inp_min,
inp_max)` -> matmul -> `clamp(out, out_min, out_max)`. The mmproj stores per-weight
`.input_min/.input_max/.output_min/.output_max` scalars.

## Tensor map (`a.blk.{il}.*`, prefix from `clip.audio`)
sscp: `a.conv1d.{0,1}.weight/bias`, `a.conv1d.{0,1}.norm.weight`, `a.input_projection.*`.
per block: `ffn_norm`, `ffn_up`, `ffn_down`, `ffn_post_norm`; `attn_pre_norm`,
`per_dim_scale`, `per_dim_k_scale`, `attn_k_rel`, q/k/v/o(+o bias), `attn_post_norm`;
`norm_conv`(=conv pre-norm, swapped name), `conv_pw1`, `conv_dw`(depthwise), `conv_norm`,
`conv_pw2`; `ffn_norm_1`, `ffn_up_1`, `ffn_down_1`, `ffn_post_norm_1`; `ln_2`.
out: `a.pre_encode.out.*`, `mm.a.soft_emb_norm.weight`, `mm.a.input_projection.weight`.

## jinfer primitive gap (why this is a large port, not a wiring job)
Present: gemm/matmul, RMS norm, SiLU/sigmoid, FlashAttention (full/causal only), Parallel.
MISSING (must be written):
1. `conv2d` stride-2 pad-1 (subsampling) + channel LayerNorm.
2. **Chunked-local attention** with overlapping-window K/V blocking (roll + strided view).
3. **Transformer-XL blocked relative-shift RPE** (+ sinusoidal pos_emb generation).
4. Attention **softcap** (tanh) path + the custom q/k scales + per-dim Q/K scaling.
5. Causal **depthwise conv1d** (ggml_ssm_conv semantics) + **GLU**.
6. **ClippableLinear** clamp wrapper on matmuls (feed per-weight min/max from the mmproj).
7. Mel STFT front-end (this file's `AudioPreprocess`).

## Recommended implementation order
1. Mel front-end (done) - verify vs llama.cpp's `log_mel_spectrogram.json` dump.
2. Subsampling + FFN + conv module + output projection (standard ops) - unit-check shapes.
3. Chunked-local attention + RPE (the parity-critical core) - build scalar/explicit first,
   validate one block's output tensor against a llama.cpp intermediate dump, THEN batch.
4. ClippableLinear clamps (get the min/max wiring right - silent parity killer if omitted).
5. Wire into `Gemma4` (route `Media.Audio` -> conformer when projector_type == gemma4a),
   E2E test the moon-landing clip for a coherent description.
Correctness before speed: a scalar-correct encode that yields coherent text is the milestone;
batched GEMMs (the "fast prefill" requirement) is a follow-up once parity holds.
