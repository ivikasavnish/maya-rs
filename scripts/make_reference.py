"""Dump deterministic reference tensors from the Python maya3 stack (CPU) for checking the Rust port.

Run with the old venv (before it is deleted):
  HF_HOME=../maya-tts/cache/huggingface ../maya-tts/.venv/bin/python scripts/make_reference.py ref.json
Greedy everywhere (cb0 = argmax without AEOS, depth argmax), so the Rust side can replay it exactly.
"""
import json
import os
import sys

import numpy as np
import torch

import maya3

ROOT = os.path.dirname(os.path.dirname(maya3.__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

from maya3 import hub  # noqa: E402
from maya3.prompt import VoiceRegistry, load_named_voices  # noqa: E402
from scripts import s2_serve_controls as S  # noqa: E402
from serve.fast_depth import FastDepth  # noqa: E402
from tts.data.packing_v2 import PackerV2  # noqa: E402
from tts.model.load_s3 import load_model_for_eval  # noqa: E402

TEXT = os.environ.get("REF_TEXT", "नमस्ते, आप कैसे हैं?")
LANG = os.environ.get("REF_LANG", "hi")
VOICE = os.environ.get("REF_VOICE", "Ahana")
FRAMES = int(os.environ.get("REF_FRAMES", "12"))

torch.set_grad_enabled(False)
r = hub.resolve("maya-research/maya2-native")
mcfg = {"model": {"backbone": str(r["backbone_dir"]), "attn": "sdpa"}, "data": {"max_tokens": 8192}}
model, _ = load_model_for_eval(str(r["ckpt"]), device="cpu", dtype=torch.bfloat16, cfg=mcfg,
                               config=os.path.join(ROOT, "configs/s3_fresh_v1.yaml"), amortize_frac=1.0,
                               require_anchor=True, verbose=False)
model.depth.float()
packer = PackerV2(model.backbone.tokenizer, model.backbone.token_ids, max_tokens=8192)
reg = VoiceRegistry()
load_named_voices(reg, r["voices_json"])
v = reg.get(VOICE)
v.prefix = None
v.spk_refs = np.zeros((1, 192), np.float32)
ex = v.builder.example(packer, v, TEXT, LANG)
prompt, n = S.prompt_from_example(ex, model, "cpu", torch.bfloat16, 32)
prompt = prompt.to(torch.bfloat16)
tokens = [int(t) for t in ex["tokens"][:n]]

mem = model.depth.spk_memory(torch.zeros(1, 1, 192), "ecapa", null_mask=torch.tensor([True]))
fd = FastDepth(model, 1, dtype=torch.float32, greedy=True, fused=False)
fd.set_speaker(0, mem, torch.ones(1, 1, dtype=torch.bool))
slots = torch.tensor([0])

out = {"text": TEXT, "lang": LANG, "voice": VOICE, "tokens": tokens,
       "prompt_last": prompt[0, -1].float().tolist(), "prompt_first": prompt[0, 0].float().tolist(), "frames": []}
h, cache = model.backbone(prompt, use_cache=True)
hl = h[:, -1]
all_codes = []
for t in range(FRAMES):
    logits = model.backbone.cb0_logits(hl).float()[0]
    cb0 = int(logits[:2048].argmax())
    codes, fb = fd._compute(slots, hl.float(), torch.tensor([cb0]))
    all_codes.append(codes[0])
    out["frames"].append({"h": hl[0].float().tolist(), "cb0_logits": logits.tolist(), "codes": codes[0].tolist(),
                          "feedback": fb[0].float().tolist()})
    h, cache = model.backbone(fb[:, None].to(torch.bfloat16), past_key_values=cache, use_cache=True)
    hl = h[:, -1]

from transformers import MimiModel  # noqa: E402

mimi = MimiModel.from_pretrained(str(r["mimi_dir"])).eval()
codes = torch.stack(all_codes, 0).T[None].long()       # [1, 32, F]
pcm = mimi.decode(codes)[0][0, 0].float()
out["pcm"] = pcm.tolist()
json.dump(out, open(sys.argv[1], "w"))
print("tokens", tokens[:8], "...", len(tokens), "pcm", pcm.shape, "codes0", all_codes[0][:8].tolist())
