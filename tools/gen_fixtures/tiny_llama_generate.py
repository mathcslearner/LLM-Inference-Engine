"""Greedy-generation goldens: HF `generate(do_sample=False)` continuations (M5-T09).

Emits tests/fixtures/models/tiny-llama/expected/generate.json — the token-for-token
oracle for `engine::engine::Generate` (docs/design/model-execution.md §10, §12).

For each fixed prompt we drive the fp32 tiny-llama (bf16 checkpoint cast to fp32 —
the same effective weights the C++ loader widens per element) through HF greedy
decoding and record the produced continuation (prompt excluded). Two independent
paths must agree before anything is written:

  * `model.generate(do_sample=False, num_beams=1)` — HF's own greedy search, the
    acceptance oracle the ticket names.
  * a hand-written prefill+decode loop over a `DynamicCache` with `argmax`
    (lowest-index tie-break) — the exact shape the C++ `Generate` runs, so the
    golden is pinned to the KV-decode path, not just batch teacher-forcing.

**EOS is suppressed** (`eos_token_id=None`) so the continuation always reaches the
full `MAX_NEW_TOKENS`, independent of whether the random tiny model happens to emit
its config EOS (id 2) by chance — the ≥32-token acceptance length is then
guaranteed. The C++ golden test calls `Generate` with an empty `eos_ids` to match;
EOS stopping is covered separately (a synthetic stop id drawn from the sequence).

`prompt + MAX_NEW_TOKENS <= max_position_embeddings` (128) for every case, so no
position runs past the model's range. Prompt ids are fixed forever — changing them
invalidates the committed golden.

This is its own subcommand (`tiny-llama-generate`) writing its own JSON file, so the
committed safetensors bytes (M5-T02/T03/T04/T05) are untouched.
"""

import pathlib

from . import common, tiny_llama

# Continuation length: comfortably above the ticket's >=32-token acceptance floor,
# and prompt+40 <= 128 for every case below.
MAX_NEW_TOKENS = 40

# (name, prompt_ids). Fixed forever. All ids < VOCAB (512), each starting with the
# BOS id (1). These prompts were selected (tools search over random prompts) for a
# **well-separated** greedy trajectory: the minimum top-2 logit gap over all
# MAX_NEW_TOKENS steps stays > 1e-2 — four orders of magnitude above the ~4e-6
# HF-vs-reference logit difference (M5-T07). That separation is what makes the
# token-for-token C++ match robust: a random tiny model has low-entropy regions
# where the top-2 logits are near-degenerate (gaps < 1e-3), and there any two
# numerically-almost-identical greedy paths (HF generate, a manual KV loop, our
# reference) flip independently — an ill-conditioned, untestable target. The
# committed activation prompt (tiny_llama.INPUT_IDS) is one such ill-conditioned
# case and is deliberately NOT reused here.
CASES = [
    ("prompt_a", [1, 348, 106, 140, 158, 283]),
    ("prompt_b", [1, 73, 279, 412, 349]),
    ("prompt_c", [1, 435, 46, 81, 209, 31, 497, 301]),
]


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tiny-llama-generate",
        help="greedy-generation goldens for the CPU reference (M5-T09)",
    )
    p.add_argument("--out", default=str(common.DEFAULT_FIXTURES_DIR))
    p.set_defaults(func=run)


def _greedy_generate(model, prompt_ids):
    """HF `generate(do_sample=False)`; returns the continuation ids (prompt excluded)."""
    import torch

    input_ids = torch.tensor([prompt_ids], dtype=torch.int64)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=None,
            pad_token_id=0,
            use_cache=True,
        )
    return out[0, len(prompt_ids):].tolist()


def _greedy_manual(model, prompt_ids):
    """Prefill+decode greedy loop over a DynamicCache; the exact shape C++ runs.

    Returns (continuation ids, min top-2 logit gap over all steps). The gap is a
    diagnostic: a comfortably large minimum gap means the fp32-order difference
    between HF and the C++ reference (max_abs_diff ~4e-6 on logits, M5-T07) cannot
    flip any argmax, so the C++ golden match is robust.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.config)
    generated = []
    min_gap = float("inf")
    cur = torch.tensor([prompt_ids], dtype=torch.int64)
    past_len = 0
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            t_len = cur.shape[1]
            positions = torch.arange(
                past_len, past_len + t_len, dtype=torch.int64
            ).unsqueeze(0)
            out = model(
                input_ids=cur,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            logits = out.logits[0, -1, :].float()
            next_id = int(torch.argmax(logits).item())
            top2 = torch.topk(logits, 2).values
            min_gap = min(min_gap, float((top2[0] - top2[1]).item()))
            generated.append(next_id)
            past_len += t_len
            cur = torch.tensor([[next_id]], dtype=torch.int64)
    return generated, min_gap


def _record_generate(out_dir) -> None:
    common.setup_torch_determinism(tiny_llama.SEED)
    model = tiny_llama._build_model().float()
    model.eval()

    cases = []
    overall_min_gap = float("inf")
    for name, prompt_ids in CASES:
        assert len(prompt_ids) + MAX_NEW_TOKENS <= tiny_llama.MAX_POS, (
            f"{name}: prompt+{MAX_NEW_TOKENS} exceeds max_position_embeddings"
        )
        assert all(0 <= i < tiny_llama.VOCAB for i in prompt_ids), f"{name}: id out of range"

        via_generate = _greedy_generate(model, prompt_ids)
        via_manual, min_gap = _greedy_manual(model, prompt_ids)
        assert via_generate == via_manual, (
            f"{name}: HF generate != manual KV loop\n"
            f"  generate: {via_generate}\n  manual:   {via_manual}"
        )
        overall_min_gap = min(overall_min_gap, min_gap)
        print(f"  {name}: {len(prompt_ids)} prompt -> {len(via_generate)} ids, min top-2 gap {min_gap:.4g}")
        cases.append({"name": name, "prompt_ids": list(prompt_ids), "generated_ids": via_generate})

    common.dump_json(
        {
            "description": "Greedy generate(do_sample=False) continuations of "
            "tiny-llama (bf16 weights cast to fp32) — the token-for-token oracle "
            "for engine::engine::Generate (M5-T09). Per case: prompt_ids [P] i32, "
            "generated_ids [max_new_tokens] i32 (prompt excluded). EOS suppressed "
            "so every case reaches max_new_tokens; the C++ golden runs with empty "
            "eos_ids. Cross-checked: HF generate == a manual prefill+decode KV "
            "loop with lowest-index argmax.",
            "eos_token_id": tiny_llama.CONFIG_JSON["eos_token_id"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_top2_logit_gap": overall_min_gap,
            "seed": tiny_llama.SEED,
            "cases": cases,
            "regenerate": "tools/.venv/bin/python -m gen_fixtures tiny-llama-generate",
            "generator_versions": common.generator_versions(),
        },
        out_dir / "generate.json",
    )


def run(args) -> None:
    out = pathlib.Path(args.out) / "models" / "tiny-llama"
    _record_generate(out / "expected")
    # Shares tiny-llama's 5 MB per-model budget; generate.json is a few KB.
    common.check_size_budget(out, 5_000_000, "models/tiny-llama")
    print(f"  wrote {out / 'expected' / 'generate.json'}")
