#!/usr/bin/env python3
"""Official references for GemmaEmbedding2ParityTest.

  f32 <in.gguf> <out.gguf>     rewrite an mmproj with every tensor in F32 (BF16/F16 widen exactly;
                               the text model goes through llama-quantize --allow-requantize ... F32)
  ref <model> <out-dir>        write cases.tsv, <case>.emb and <case>.proj from the official pipeline:
                               sentence-transformers (>= 6.1) over google/embeddinggemma-2 (a hub id
                               or a local snapshot) in float32 on CPU

f32 needs llama.cpp's gguf-py on PYTHONPATH; ref needs torch, transformers (with embedding_gemma2),
sentence-transformers, pillow, numpy, scipy and soundfile.
"""
import os, struct, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IMAGES = os.path.join(ROOT, 'jinfer', 'jinfer-langchain4j', 'src', 'test', 'resources', 'kit-images')
AUDIO = os.path.join(ROOT, 'test-fixtures', 'audio')


def f32(src, dst):
    import numpy as np
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T, GGUFValueType as V
    r = GGUFReader(src)
    w = GGUFWriter(dst, r.fields['general.architecture'].contents())
    for k, f in r.fields.items():
        if k.startswith('GGUF.') or k == 'general.architecture':
            continue
        if k == 'general.file_type':
            w.add_uint32(k, 0)
        elif f.types[0] == V.ARRAY:
            w.add_array(k, f.contents())
        else:
            w.add_key_value(k, f.contents(), f.types[0])
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape)]
        if t.tensor_type == T.BF16:
            a = (t.data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif t.tensor_type in (T.F16, T.F32):
            a = np.asarray(t.data).astype(np.float32)
        else:
            raise SystemExit(f'{t.name}: {t.tensor_type.name} is not a float tensor')
        w.add_tensor(t.name, np.ascontiguousarray(a.reshape(shape)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def pattern(path, width, height, shift=0):
    """A synthetic image, PNG: smooth waves and fine stripes, drifting right as shift grows."""
    import numpy as np
    from PIL import Image
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([127 + 120 * np.sin((x + 40 * shift) / 23.0) * np.cos(y / 31.0),
                    127 + 120 * np.sin((x + y + 25 * shift) / 17.0), (x * 7 + y * 3 + 50 * shift) % 256], -1)
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return path


def wav16k(src, dst, seconds=None):
    """Mono 16 kHz s16, what the model card asks for; both sides then read the same samples."""
    import numpy as np, scipy.signal, soundfile
    pcm, rate = soundfile.read(src, dtype='float32')
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if rate != 16000:
        pcm = scipy.signal.resample_poly(pcm, 16000, rate)
    if seconds:
        pcm = pcm[:seconds * 16000]
    soundfile.write(dst, np.clip(pcm, -1, 32767 / 32768), 16000, subtype='PCM_16')
    return dst


def ref(model_id, out):
    import numpy as np, soundfile, torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    os.makedirs(out, exist_ok=True)
    aligned = pattern(os.path.join(out, 'aligned.png'), 576, 432)  # a 12x9 grid of 48-pixel tokens
    cat, dice = os.path.join(IMAGES, 'cat.png'), os.path.join(IMAGES, 'dice.png')
    speech = wav16k(os.path.join(AUDIO, 'speech-tts.wav'), os.path.join(out, 'speech-16k.wav'))
    long_audio = wav16k(os.path.join(AUDIO, 'long-35s.wav'), os.path.join(out, 'long-30s.wav'), 30)
    # three frames one second apart, joined by '+'; frame choice is VideoCodec.every's test
    video = '+'.join(pattern(os.path.join(out, f'frame{f}.png'), 640, 360, f) for f in range(3))
    long_text = ' '.join(f'Sentence number {i} talks about the northern lights, auroras and solar wind.'
                         for i in range(120))  # 2058 tokens: past the 1024-wide window
    cases = {
        'query': ('task: search result | query: What causes the northern lights?', []),
        'document': ('title: none | text: The northern lights are caused by charged particles from the sun.', []),
        'long': ('title: none | text: ' + long_text, []),
        'aligned': ('{m}', [aligned]),
        'cat': ('{m}', [cat]),  # grows to the 280-token budget
        'dice': ('{m}', [dice]),
        'cat_text': ('A photo of a cat: {m} It is sitting.', [cat]),
        'speech': ('{m}', [speech]),
        'chirp': ('{m}', [os.path.join(AUDIO, 'chirp-5s.wav')]),
        'long_audio': ('{m}', [long_audio]),  # 30 s: the reference's feature extractor cuts there
        'mixed': ('task: search result | query: {m} and {m}', [cat, speech]),
        'video': ('{m}', [video]),  # frames at the video budget, 140 tokens: 720x384
        'video_text': ('A pattern drifting right: {m}', [video]),
    }
    with open(os.path.join(out, 'cases.tsv'), 'w') as f:
        f.writelines(f'{n}\t{t}\t{",".join(m)}\n' for n, (t, m) in cases.items())

    st = SentenceTransformer(model_id, device='cpu', model_kwargs={'dtype': torch.float32})
    model = st[0].model

    def load(path):
        if '+' in path:
            return 'video', '<|video|>', np.stack([np.asarray(Image.open(p).convert('RGB')) for p in path.split('+')])
        if path.endswith('.wav'):
            pcm, rate = soundfile.read(path, dtype='float32')
            assert rate == 16000, path
            return 'audio', '<|audio|>', {'array': pcm, 'sampling_rate': 16000}
        return 'image', '<|image|>', Image.open(path).convert('RGB')

    def write(path, rows):
        rows = np.asarray(rows, dtype='<f4')
        with open(path, 'wb') as f:
            f.write(struct.pack('<ii', *rows.shape) + rows.tobytes())

    with torch.no_grad():
        for name, (text, media) in cases.items():
            item, loaded = {'text': text}, [load(m) for m in media]
            for kind, marker, value in loaded:
                item['text'] = item['text'].replace('{m}', marker, 1)
                item.setdefault(kind, []).append(value)
            item = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in item.items()}
            embedding = st.encode([item] if media else [text], convert_to_numpy=True)[0]
            with open(os.path.join(out, name + '.emb'), 'wb') as f:
                f.write(np.asarray(embedding, dtype='<f4').tobytes())
            kind, _, value = loaded[0] if len(media) == 1 else (None, None, None)
            if kind in ('image', 'audio'):  # the projector's rows, as the text model receives them
                if kind == 'image':
                    p = st[0].processor.image_processor([value], return_tensors='pt')
                    rows = model.get_image_features(p['pixel_values'], p['image_position_ids']).pooler_output[0]
                else:
                    f = st[0].processor.feature_extractor([value['array']], sampling_rate=16000, return_tensors='pt')
                    a = model.get_audio_features(f['input_features'], f['input_features_mask'])
                    rows = a.pooler_output[a.attention_mask]
                write(os.path.join(out, name + '.proj'), rows.numpy())
            print(name, len(embedding))


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'f32':
        f32(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == 'ref':
        ref(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
