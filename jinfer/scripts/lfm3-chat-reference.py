#!/usr/bin/env python3
"""Greedy llama-server generations for Lfm3GenerationParityTest.

  lfm3-chat-reference.py <text|vl|omni> <port> <out.json>

Start llama-server on the matching checkpoint first, e.g.
  llama-server -m LFM3-Omni-2B-Q8_0.gguf --mmproj mmproj-LFM3-Omni-2B-F16.gguf --jinja --port 8099

Per case: the prompt the server renders (/apply-template), as token ids split at its media markers
(mtmd frames each one as <|vision_start|> rows <|vision_end|>), then a temperature-0 /completion
with the raw generated ids, each step's top-2 logprobs, and the positions the prompt took.
"""
import base64, json, os, sys, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'test-fixtures'))

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def media(kind, path):
    with open(os.path.join(ROOT, path), 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    if kind == 'image':
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}, data
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}, data


def text(t):
    return {"type": "text", "text": t}


# name -> (parts of the one user turn, or a full message list), tools, thinking, max tokens
CASES = {
    'capital': dict(user=["What is the capital of France? Answer in one word."]),
    'haiku': dict(system="You are a poet.", user=["Write a haiku about autumn rain."]),
    'multiply': dict(user=["What is 17 * 23?"], thinking=True, max_tokens=160),
    'weather': dict(user=["What's the weather in Paris?"], tools=[WEATHER]),
}
MEDIA = {
    'vl': {
        'cat': dict(user=[('image', 'lfm3/images/cat.png'), "Describe this image in one sentence."]),
        'dice': dict(user=["How many dice are in this picture?", ('image', 'lfm3/images/dice.png')]),
    },
    'omni': {
        'cat': dict(user=[('image', 'lfm3/images/cat.png'), "Describe this image in one sentence."]),
        'transcribe': dict(user=[('audio', 'parakeet/ls0.wav'), "Transcribe the audio."]),
        'listen': dict(user=["What is said in this recording?", ('audio', 'parakeet/en0.wav')]),
    },
}


def post(port, path, body):
    request = urllib.request.Request(f'http://localhost:{port}{path}', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(request))


def main(kind, port, out):
    cases = dict(CASES)
    cases.update(MEDIA.get(kind, {}))
    marker = json.load(urllib.request.urlopen(f'http://localhost:{port}/props')).get('media_marker', '<__media__>')
    results = {}
    for name, case in cases.items():
        parts, blobs = [], []
        for part in case['user']:
            if isinstance(part, tuple):
                content, data = media(*part)
                parts.append(content)
                blobs.append(data)
            else:
                parts.append(text(part))
        messages = ([{"role": "system", "content": case['system']}] if 'system' in case else []) + \
                   [{"role": "user", "content": parts if blobs else parts[0]['text']}]
        body = {"messages": messages, "chat_template_kwargs": {"enable_thinking": case.get('thinking', False)}}
        if 'tools' in case:
            body['tools'] = case['tools']
        prompt = post(port, '/apply-template', body)['prompt']
        pieces = [post(port, '/tokenize', {"content": piece, "parse_special": True})['tokens']
                  for piece in prompt.split(marker)]
        completion = post(port, '/completion', {
            "prompt": {"prompt_string": prompt, "multimodal_data": blobs} if blobs else prompt,
            # greedy for real: the GGUF's general.sampling.penalty_repeat would otherwise apply
            "temperature": 0, "repeat_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "dry_multiplier": 0,
            "n_predict": case.get('max_tokens', 64), "return_tokens": True, "n_probs": 2,
            "cache_prompt": False})
        results[name] = {
            "pieces": pieces, "media": [list(p) for p in case['user'] if isinstance(p, tuple)],
            "thinking": case.get('thinking', False), "system": case.get('system'),
            "user": [p if isinstance(p, str) else None for p in case['user']], "tools": 'tools' in case,
            "prompt_positions": completion['tokens_evaluated'], "tokens": completion['tokens'],
            "top2": [[t['logprob'] for t in p['top_logprobs']] for p in completion['completion_probabilities']],
            "text": completion['content']}
        print(name, completion['tokens_evaluated'], json.dumps(completion['content'])[:160])
    with open(out, 'w') as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main(*sys.argv[1:])
