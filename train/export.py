"""Quantize a checkpoint and write the files the device needs.

    python train/export.py

    python train/export.py --from-bin firmware/data/model.bin   # same weights,
                                   # rebuilt index, keys and errata (no checkpoint
                                   # or torch needed: after editing phrasings, or
                                   # when the file layout changes)

Writes:
    firmware/data/model.bin        the model; the host build loads it and the firmware
                                   embeds it in flash (firmware/embed_model.py)
    firmware/data/model.tier       its fact set (small, max4mb, large)
    firmware/data/model.dropped    keys of facts it refuses for lack of room (--max-bytes)

File layout, version 4 (int4 weights), 5 (ternary) or 6 (int2), little endian,
every block padded to 4 bytes:
    u32 magic "TAI3", version, vocab, ctx, dim, layers, heads, kv_heads,
        hidden, n_facts, n_known
    tok, pos      int8 [rows*cols] + f32 [rows] scales
    per layer:    n1 f32[dim], wq wk wv wo, n2 f32[dim], w1 w2 (quantized)
                  int4 = [rows*cols/2] bytes, two weights per byte, low
                  nibble first, stored +8; then f16 [rows*cols/32] scales
                  int2 = [rows*cols/4] bytes, four weights per byte, first in
                  the lowest 2 bits, stored +2 (value = stored - 1.5); then
                  f16 [rows*cols/32] scales
                  ternary = [rows*ceil(cols/5)] bytes, five weights per byte
                  in base 3, first in the lowest digit, stored +1; then
                  f16 [rows] scales
    norm          f32[dim]
    gate index    see train/gate_index.py: word dictionary (u32 base per 256
                  words + u16 offsets), phrasings grouped by fact (u8 count per
                  fact), per phrasing its question type (+0x80: every word must
                  match; +0x40: one extra question word is allowed) and length,
                  then all phrasings' word ids in a row (+0x8000 on each
                  phrasing's last; 0xFFFF for a phrasing with no words)
    keys          each fact's canonical key (shortest phrasing), coded: u32
                  n_extra, u16 offsets and a u32-sized blob of extra words, then
                  u32 bytes of codes (see encode_keys)
    errata        u32 count, u16 fact indices (sorted), then their answers as
                  strings: facts the quantized model answers wrong, found by
                  running the C engine (host/tinyai) on every fact's key. With
                  --max-bytes, the ones that don't fit get an empty answer: the
                  engine refuses them, and their key is stored empty
"""

import argparse
import hashlib
import os
import struct
import sys

try:  # not needed with --from-bin
    import numpy as np
    import torch

    from model import GROUP, quantize_int2, quantize_int4, quantize_ternary
except ImportError:
    torch = None

import engine
import gate_index
from common import canonical, load_facts, normalize, tier_of


def pad4(b):
    return b + b"\0" * (-len(b) % 4)


def q8(w):
    w = w.detach().float().numpy()
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return pad4(q.tobytes()) + scale.astype("<f4").tobytes()


def q4(w):
    """int4 with the exact rounding QAT trained against."""
    rows, cols = w.shape
    g = w.detach().float().reshape(rows, cols // GROUP, GROUP)
    scale = (g.abs().amax(-1, keepdim=True) / 7).clamp(min=6.2e-5).half().float()
    q = torch.clamp(torch.round(g / scale), -7, 7).to(torch.int16).reshape(rows, cols)
    assert torch.equal((q.reshape(rows, cols // GROUP, GROUP) * scale).reshape(rows, cols),
                       quantize_int4(w.detach().float()))
    n = (q + 8).numpy().astype(np.uint8)
    packed = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8)
    return pad4(packed.tobytes()) + pad4(scale.reshape(-1).numpy().astype("<f2").tobytes())


def q2(w):
    """int2 with the exact rounding QAT trained against: 4 weights per byte,
    first in the lowest 2 bits, stored +2; then one fp16 scale per 32."""
    rows, cols = w.shape
    g = w.detach().float().reshape(rows, cols // GROUP, GROUP)
    scale = (g.abs().mean(-1, keepdim=True) * 1.2).clamp(min=6.2e-5).half().float()
    q = torch.clamp(torch.floor(g / scale), -2, 1)
    assert torch.equal(((q + 0.5) * scale).reshape(rows, cols), quantize_int2(w.detach().float()))
    u = (q + 2).to(torch.uint8).reshape(rows, cols // 4, 4).numpy()
    packed = (u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)).astype(np.uint8)
    return pad4(packed.tobytes()) + pad4(scale.reshape(-1).numpy().astype("<f2").tobytes())


def qt(w):
    """Ternary: 5 weights per byte in base 3 (first weight in the lowest digit,
    stored +1), each row padded to whole bytes, then one fp16 scale per row."""
    rows, cols = w.shape
    w = w.detach().float()
    scale = w.abs().mean(-1, keepdim=True).clamp(min=6.2e-5).half().float()
    t = torch.clamp(torch.round(w / scale), -1, 1)
    assert torch.equal(t * scale, quantize_ternary(w))
    stride = -(-cols // 5)
    u = np.zeros((rows, stride * 5), np.uint8)
    u[:, :cols] = (t + 1).numpy().astype(np.uint8)
    u = u.reshape(rows, stride, 5).astype(np.uint16)
    packed = (u[..., 0] + 3 * u[..., 1] + 9 * u[..., 2] + 27 * u[..., 3] + 81 * u[..., 4]).astype(np.uint8)
    return pad4(packed.tobytes()) + pad4(scale.reshape(-1).numpy().astype("<f2").tobytes())


def f32(t):
    return t.detach().float().numpy().astype("<f4").tobytes()


def strings(items):
    blob = "\0".join(items).encode("ascii") + b"\0"
    return struct.pack("<I", len(blob)) + pad4(blob)


def encode_keys(keys, wid):
    """Keys (each fact's shortest phrasing, the model's prompt) as codes: a byte
    below 0x80 is a literal character; 0x80 | hi, lo is word hi << 8 | lo, from
    the gate's dictionary or, after it, from `extras` (other frequent words).
    A word gets a space before it unless it starts the key. Keys end in 0."""
    count = {}
    for k in keys:
        for t in k.split(" "):
            if t not in wid:
                count[t] = count.get(t, 0) + 1
    # a word worth coding: saves (len - 2) bytes each time, costs len + 3 once
    extras = sorted((t for t, n in count.items() if n * (len(t) - 2) > len(t) + 3),
                    key=lambda t: -count[t] * (len(t) - 2))
    code = dict(wid)
    for t in extras:
        code[t] = len(code)
    assert len(code) < 32768
    out = bytearray()
    for k in keys:
        for i, t in enumerate(k.split(" ")):
            c = code.get(t)
            if c is not None and len(t) > 2:
                out += bytes([0x80 | c >> 8, c & 0xFF])
            else:
                out += (b" " if i else b"") + t.encode("ascii")
        out += b"\0"
    return extras, bytes(out)


def decode_keys(c, wid, extras):
    """Mirror of tai_fact in tinyai.c, to check the codes round-trip. A code's
    second byte can be 0, so keys are parsed, not split on 0."""
    words = sorted(wid, key=wid.get) + extras
    keys, s, i = [], "", 0
    while i < len(c):
        if c[i] >= 0x80:
            s += (" " if s else "") + words[(c[i] & 0x7F) << 8 | c[i + 1]]
            i += 2
        elif c[i]:
            s += chr(c[i])
            i += 1
        else:
            keys.append(s)
            s, i = "", i + 1
    return keys


def pack_keys(keys, wid):
    """The keys block: extra words (offsets + strings), then the coded keys."""
    extras, codes = encode_keys(keys, wid)
    assert decode_keys(codes, wid, extras) == keys
    ext = "\0".join(extras).encode("ascii") + b"\0"
    offs, o = [], 0
    for e in extras:
        offs.append(o)
        o += len(e) + 1
    assert o < 65536
    return (struct.pack("<I", len(extras)) + pad4(struct.pack(f"<{len(extras)}H", *offs)) +
            struct.pack("<I", len(ext)) + pad4(ext) + struct.pack("<I", len(codes)) + pad4(codes))


def fit(wrong, strict, assemble, max_bytes):
    """Errata for the facts the model gets wrong: [(fact, answer)], sorted.
    Without a size limit every answer is stored. With one, answers are stored
    in priority order (hand-written and survival facts first and always, then
    Wikidata facts, best known first) and the rest get "" (refused)."""
    if not max_bytes or len(assemble(wrong)) <= max_bytes:
        return wrong
    order = [w for w in wrong if w[0] not in strict] + [w for w in wrong if w[0] in strict]

    def plan(k):  # store the first k answers of `order`, refuse the rest
        return sorted(order[:k] + [(i, "") for i, _ in order[k:]])

    must = sum(i not in strict for i, _ in wrong)
    if len(assemble(plan(must))) > max_bytes:
        sys.exit(f"hand-written facts alone need {len(assemble(plan(must))):,} bytes, over {max_bytes:,}")
    lo, hi = must, len(order)  # plan(lo) fits, plan(hi) doesn't
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if len(assemble(plan(mid))) <= max_bytes else (lo, mid)
    return plan(lo)


def errata(fixes):
    """[(fact index, answer)] -> the errata block (see the layout above)."""
    return (struct.pack("<I", len(fixes)) + pad4(struct.pack(f"<{len(fixes)}H", *(i for i, _ in fixes))) +
            strings([a for _, a in fixes]))


ask = engine.ask  # the C engine, on all CPU cores


def write(args, data, dropped):
    with open(args.bin, "wb") as f:
        f.write(data)
    stem = os.path.splitext(args.bin)[0]
    with open(stem + ".tier", "w") as f:
        f.write(args.tier + "\n")  # which fact set this model knows (read by the tests)
    with open(stem + ".dropped", "w") as f:  # facts refused for lack of room (read by the tests)
        f.write("".join(k + "\n" for k in dropped))
    if not args.header:
        return
    with open(args.header, "w") as f:
        f.write("// Generated by train/export.py. Do not edit.\n#pragma once\n#include <stdint.h>\n\n")
        f.write(f"static const unsigned int model_data_len = {len(data)};\n")
        f.write("static const uint8_t model_data[] __attribute__((aligned(4))) = {\n")
        for i in range(0, len(data), 32):
            f.write(",".join(str(x) for x in data[i:i + 32]) + ",\n")
        f.write("};\n")


MAGIC = 0x33494154  # "TAI3"
VERSIONS = {"int4": 4, "ternary": 5, "int2": 6}
GROUP_ = 32  # weights per scale (model.GROUP), without importing torch


def quantized_weights(c, sd):
    """(file version, bytes of everything from tok to norm) for a checkpoint."""
    kind = c.get("weights", "int4")
    qw = {"int4": q4, "ternary": qt, "int2": q2}[kind]
    out = [q8(sd["tok.weight"]), q8(sd["pos.weight"])]
    for l in range(c["layers"]):
        p = f"blocks.{l}."
        out.append(f32(sd[p + "n1.w"]))
        out += [qw(sd[p + n + ".weight"]) for n in ("wq", "wk", "wv", "wo")]
        out.append(f32(sd[p + "n2.w"]))
        out += [qw(sd[p + n + ".weight"]) for n in ("w1", "w2")]
    out.append(f32(sd["norm.w"]))
    return VERSIONS[kind], b"".join(out)


def weights_from_bin(path):
    """(version, cfg, weight bytes, parameter count) of an exported model file,
    of this or the previous ("TAI2") layout: the weights are the same."""
    data = open(path, "rb").read()
    magic, version, vocab, ctx, dim, layers, heads, kv_heads, hidden = struct.unpack("<9I", data[:36])
    if magic not in (MAGIC, 0x32494154) or version not in VERSIONS.values():
        sys.exit(f"{path}: not a model file")

    def p4(n):
        return -(-n // 4) * 4

    def q(rows, cols):  # one quantized matrix, as q4 / q2 / qt write it
        if version == 6:
            return p4(rows * cols // 4) + p4(rows * cols // GROUP_ * 2)
        if version == 5:
            return p4(rows * -(-cols // 5)) + p4(rows * 2)
        return p4(rows * cols // 2) + p4(rows * cols // GROUP_ * 2)

    kv = kv_heads * (dim // heads)
    size = (p4(vocab * dim) + 4 * vocab + p4(ctx * dim) + 4 * ctx + 4 * dim +
            layers * (8 * dim + 2 * q(dim, dim) + 2 * q(kv, dim) + q(hidden, dim) + q(dim, hidden)))
    c = dict(vocab=vocab, ctx=ctx, dim=dim, layers=layers, heads=heads, kv_heads=kv_heads, hidden=hidden)
    n_params = (vocab + ctx) * dim + layers * (2 * dim + 2 * dim * dim + 2 * kv * dim + 2 * hidden * dim) + dim
    return version, c, data[44:44 + size], n_params


def model_answers(args, keys, weights, batch=2000):
    """The model's own answer to each key (the file at args.bin has no errata
    yet). Saved as it goes in <bin>.answers, keyed by a hash of the weights and
    the engine's code, so an interrupted export resumes and a re-export with
    the same weights (new phrasings, a new file layout) runs no model at all."""
    here = os.path.dirname(os.path.abspath(__file__))
    engine_src = open(os.path.join(here, "..", "firmware", "lib", "tinyai", "tinyai.c"), "rb").read()
    tag = hashlib.sha1(weights + engine_src).hexdigest()
    path = os.path.splitext(args.bin)[0] + ".answers"
    cache = {}
    if os.path.exists(path):
        lines = open(path, encoding="utf-8").read().split("\n")
        if lines[0] == tag:
            cache = dict(l.split("\t", 1) for l in lines[1:] if "\t" in l)
    if not cache:
        with open(path, "w", encoding="utf-8") as f:
            f.write(tag + "\n")
    todo = [k for k in dict.fromkeys(keys) if k not in cache]
    print(f"model answers: {len(keys) - len(todo)} cached, {len(todo)} to run", flush=True)
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        got = ask(args.host, args.bin, part)
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(f"{k}\t{g}\n" for k, g in zip(part, got)))
        cache.update(zip(part, got))
        print(f"  {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return [cache[k] for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, choices=["small", "max4mb", "large"],
                    help="fact set the checkpoint was trained on (default: recorded in it by train.py)")
    ap.add_argument("--ckpt", default="train/ckpt.pt")
    ap.add_argument("--from-bin", default=None,
                    help="take the weights from this model file instead of a checkpoint")
    ap.add_argument("--bin", default="firmware/data/model.bin")
    ap.add_argument("--header", default=None, help="also write the model as a C array (optional; the firmware embeds the .bin)")
    ap.add_argument("--host", default="host/tinyai", help="C engine used to find errata")
    ap.add_argument("--no-check", action="store_true", help="skip the errata check")
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="size limit: facts the model gets wrong that don't fit as errata are refused")
    args = ap.parse_args()

    if args.from_bin:
        version, c, weights, n_params = weights_from_bin(args.from_bin)
        args.tier = args.tier or tier_of(args.from_bin)
    else:
        if torch is None:
            sys.exit("exporting a checkpoint needs torch and numpy (pip install -r requirements.txt)")
        ck = torch.load(args.ckpt, map_location="cpu")
        c, sd = ck["cfg"], ck["model"]
        if args.tier and ck.get("tier") and args.tier != ck["tier"]:
            sys.exit(f"{args.ckpt} was trained on tier {ck['tier']!r}, not {args.tier!r}")
        args.tier = args.tier or ck.get("tier")
        if not args.tier:
            sys.exit(f"{args.ckpt} does not record its tier (older train.py): pass --tier")
        version, weights = quantized_weights(c, sd)
        n_params = sum(v.numel() for v in sd.values())
    strict, lenient = set(), set()
    facts = load_facts(args.tier, strict=strict, lenient=lenient)
    keys = [canonical(qs) for qs, _ in facts]
    known = {}
    for i, (qs, _) in enumerate(facts):
        for q in qs:
            known.setdefault(normalize(q), i)  # first fact wins a shared phrasing
    assert len(facts) < 65536

    out = [struct.pack("<11I", MAGIC, version, c["vocab"], c["ctx"], c["dim"], c["layers"],
                       c["heads"], c["kv_heads"], c["hidden"], len(facts), len(known)), weights]
    index, wid = gate_index.build(list(known.items()), len(facts), strict, lenient)
    out.append(index)
    n_words = len(wid)
    base = b"".join(out)

    def assemble(fixes):
        """The whole file. A fact with an empty erratum is refused, so its key
        (only ever the model's prompt) is stored empty."""
        drop = {i for i, a in fixes if not a}
        return base + pack_keys([("" if i in drop else k) for i, k in enumerate(keys)], wid) + errata(fixes)

    data = assemble([])
    write(args, data, [])
    fixes = []
    if not args.no_check:
        if not os.path.exists(args.host):
            sys.exit(f"{args.host} not found: build it with `make -C host`, or pass --no-check")
        # Errata are looked up by the fact the gate picks, so every key must
        # reach its own fact (check_routing.py checks all other phrasings).
        routes = ask(args.host, args.bin, keys, "--gate")
        astray = [(k, r) for k, r, (_, a) in zip(keys, routes, facts) if r != k and r != "=" + a]
        assert not astray, f"keys that don't reach their own fact: {astray[:5]}"
        # the real C engine answers every fact's key, exactly as on the device
        got = model_answers(args, keys, weights)
        wrong = [(i, a) for i, ((_, a), g) in enumerate(zip(facts, got)) if g != a]
        fixes = fit(wrong, strict, assemble, args.max_bytes)
        if fixes:
            data = assemble(fixes)
            write(args, data, [keys[i] for i, a in fixes if not a])
            # a stored answer is now exact; a dropped fact is refused, never wrong
            still = [i for (i, a), g in zip(fixes, ask(args.host, args.bin, [keys[i] for i, _ in fixes]))
                     if g != (a or "I don't know.")]
            assert not still, f"errata did not take: {[keys[i] for i in still[:5]]}"
        dropped = sum(not a for _, a in fixes)
        print(f"C engine: {len(facts) - len(wrong)}/{len(facts)} facts exact from the model; "
              f"{len(wrong) - dropped} stored as errata, {dropped} refused (no room)")
        for i, a in fixes[:20]:
            print(f"    {keys[i]!r} -> {got[i]!r}, {'stored ' + repr(a) if a else 'refused'}")
    if args.max_bytes and len(data) > args.max_bytes:
        sys.exit(f"{len(data):,} bytes, over --max-bytes {args.max_bytes:,}")
    print(f"{n_params:,} params -> {len(data):,} bytes ({8 * len(data) / n_params:.2f} bits/param "
          f"incl. text), {len(facts)} facts, {len(known)} phrasings, {n_words} indexed words, "
          f"gate index {len(index):,} bytes")


if __name__ == "__main__":
    main()
