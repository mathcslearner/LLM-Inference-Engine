"""Greedy-generation goldens for tiny-qwen2: HF `generate(do_sample=False)` (M5-T10).

Emits tests/fixtures/models/tiny-qwen2/expected/generate.json — the token-for-token
oracle for `engine::engine::Generate` on the Qwen fixture (docs/design/
model-execution.md §10, §12). The Qwen-family analog of tiny_llama_generate.py; see
that file for the full rationale on why two independent decode paths must agree and
why the prompts are chosen for a well-separated greedy trajectory.

For each fixed prompt we drive the fp32 tiny-qwen2 (bf16 checkpoint cast to fp32 —
the same effective weights the C++ loader widens per element) through HF greedy
decoding and record the continuation (prompt excluded). Two paths must agree before
anything is written:

  * `model.generate(do_sample=False, num_beams=1)` — HF's greedy search.
  * a hand-written prefill+decode loop over a `DynamicCache` with `argmax`
    (lowest-index tie-break) — the exact shape C++ `Generate` runs.

**EOS is suppressed** (`eos_token_id=None`) so every continuation reaches the full
`MAX_NEW_TOKENS`; the C++ golden test calls `Generate` with an empty `eos_ids`.

Unlike a BOS-prefixed Llama prompt, Qwen inserts no BOS, so these prompt id streams
deliberately do not begin with any special id. `prompt + MAX_NEW_TOKENS <=
max_position_embeddings` (128) for every case. Prompt ids are fixed forever.

This is its own subcommand (`tiny-qwen2-generate`) writing its own JSON file, so the
committed tiny-qwen2 safetensors bytes are untouched.
"""

import pathlib

from . import common, tiny_qwen2

# Continuation length: comfortably above the >=32-token acceptance floor, and
# prompt+40 <= 128 for every case below.
MAX_NEW_TOKENS = 40

# The minimum top-2 logit gap (over every step of every case) must exceed this for
# the token-for-token C++ match to be robust: it is four orders of magnitude above
# the ~1e-5 HF-vs-reference logit difference, so no fp32-order divergence can flip an
# argmax. Enforced as an assertion below, not merely reported (a tightening over
# M5-T09, where it was diagnostic only).
MIN_TOP2_GAP = 1e-2

# (name, prompt_ids). Fixed forever. All ids < VOCAB (512), no BOS. Selected (an
# offline search over random prompts against this exact model) so the minimum top-2
# logit gap across all MAX_NEW_TOKENS steps stays > MIN_TOP2_GAP — see the module
# docstring and tiny_llama_generate.py for why ill-conditioned prompts are unusable.
CASES = [
    ("prompt_a", [174, 210, 487, 51]),
    ("prompt_b", [317, 307, 105, 432, 68]),
    ("prompt_c", [260, 343, 420, 102, 247]),
]


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "tiny-qwen2-generate",
        help="greedy-generation goldens for the Qwen fixture (M5-T10)",
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

    Returns (continuation ids, min top-2 logit gap over all steps).
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
    common.setup_torch_determinism(tiny_qwen2.SEED)
    model = tiny_qwen2._build_model().float()
    model.eval()

    cases = []
    overall_min_gap = float("inf")
    for name, prompt_ids in CASES:
        assert len(prompt_ids) + MAX_NEW_TOKENS <= tiny_qwen2.MAX_POS, (
            f"{name}: prompt+{MAX_NEW_TOKENS} exceeds max_position_embeddings"
        )
        assert all(0 <= i < tiny_qwen2.VOCAB for i in prompt_ids), f"{name}: id out of range"

        via_generate = _greedy_generate(model, prompt_ids)
        via_manual, min_gap = _greedy_manual(model, prompt_ids)
        assert via_generate == via_manual, (
            f"{name}: HF generate != manual KV loop\n"
            f"  generate: {via_generate}\n  manual:   {via_manual}"
        )
        overall_min_gap = min(overall_min_gap, min_gap)
        print(f"  {name}: {len(prompt_ids)} prompt -> {len(via_generate)} ids, min top-2 gap {min_gap:.4g}")
        cases.append({"name": name, "prompt_ids": list(prompt_ids), "generated_ids": via_generate})

    assert overall_min_gap > MIN_TOP2_GAP, (
        f"min top-2 logit gap {overall_min_gap:.4g} <= {MIN_TOP2_GAP}: the greedy "
        f"trajectory is ill-conditioned — pick better-separated prompts (see docstring)"
    )

    common.dump_json(
        {
            "description": "Greedy generate(do_sample=False) continuations of "
            "tiny-qwen2 (bf16 weights cast to fp32) — the token-for-token oracle "
            "for engine::engine::Generate on the Qwen fixture (M5-T10). Per case: "
            "prompt_ids [P] i32 (no BOS), generated_ids [max_new_tokens] i32 (prompt "
            "excluded). EOS suppressed so every case reaches max_new_tokens; the C++ "
            "golden runs with empty eos_ids. Cross-checked: HF generate == a manual "
            "prefill+decode KV loop with lowest-index argmax.",
            "eos_token_id": tiny_qwen2.CONFIG_JSON["eos_token_id"],
            "max_new_tokens": MAX_NEW_TOKENS,
            "min_top2_logit_gap": overall_min_gap,
            "seed": tiny_qwen2.SEED,
            "cases": cases,
            "regenerate": "tools/.venv/bin/python -m gen_fixtures tiny-qwen2-generate",
            "generator_versions": common.generator_versions(),
        },
        out_dir / "generate.json",
    )


def run(args) -> None:
    out = pathlib.Path(args.out) / "models" / "tiny-qwen2"
    _record_generate(out / "expected")
    # Shares tiny-qwen2's 5 MB per-model budget; generate.json is a few KB.
    common.check_size_budget(out, 5_000_000, "models/tiny-qwen2")
    print(f"  wrote {out / 'expected' / 'generate.json'}")
