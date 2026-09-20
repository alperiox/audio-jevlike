# src/prosodia/model/prosodia.py
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from prosodia.model.branches import IsolatedBranches
from prosodia.model.heads import ReadoutHead
from prosodia.model.qencoder import QuestionEncoder
from prosodia.model.state import StateEncoder


class ProsodiaModel(nn.Module):
    """state (audio + context) -> per-question logits over supplied options."""

    def __init__(
        self, in_dim: int, d_model: int = 256, state_layers: int = 2,
        branch_layers: int = 2, n_heads: int = 4, stride: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_encoder = StateEncoder(in_dim, d_model, state_layers, n_heads, stride)
        self.question_encoder = QuestionEncoder(d_model=d_model)
        self.branches = IsolatedBranches(d_model, branch_layers, n_heads)
        self.readout = ReadoutHead(d_model)
        self.audio_absent = nn.Parameter(torch.zeros(d_model))
        self.context_absent = nn.Parameter(torch.zeros(d_model))

    def _encode_state(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        h, mask = self.state_encoder(batch["audio"], batch["audio_mask"])

        # Modality dropout: replace the whole audio state with a learned token.
        audio_present = batch["audio_present"].to(h.device).view(-1, 1, 1)
        h = torch.where(audio_present, h, self.audio_absent.view(1, 1, -1).expand_as(h))

        # Canonicalize the mask alongside the content. Content-neutralizing `h`
        # alone is not enough: `mask` still carries the REAL per-example audio
        # duration (pooled frames + the two C2 stat-token positions, both
        # audio-derived), and that duration reaches `IsolatedBranches`'
        # `nn.MultiheadAttention` as `key_padding_mask`. Even with every muted
        # position holding the identical `audio_absent` vector, the softmax
        # weight that whole block receives is a function of how MANY valid
        # positions there are -- i.e. of the real, un-muted audio length. That
        # is a second, larger side channel duration leaks through, independent
        # of the content gate above. Zeroing the mask here removes the muted
        # positions from attention entirely rather than merely neutralizing
        # their content, so a muted row carries no information about audio at
        # all -- not content, not duration.
        mask = mask & audio_present.view(-1, 1)

        ctx_present = batch["context_present"].to(h.device)
        ctx_vecs = self.question_encoder.embed_texts(list(batch["context"]))
        ctx_vecs = torch.where(ctx_present.view(-1, 1), ctx_vecs,
                               self.context_absent.view(1, -1).expand_as(ctx_vecs))

        # Context joins the state as one extra position the branches attend to.
        h = torch.cat([ctx_vecs.unsqueeze(1), h], dim=1)
        mask = torch.cat([torch.ones(h.shape[0], 1, dtype=torch.bool, device=mask.device),
                          mask], dim=1)
        return h, mask

    def forward(self, batch: dict[str, Any]) -> list[dict[str, Tensor]]:
        """Batched version of `_forward_looped_reference` (I6 fix).

        `collate_batch` builds a padded `(B, T, D)` batch and `_encode_state`
        runs it through in one shot, but the reference implementation this
        replaces immediately un-batched that work: it called
        `self.branches` with `b=1` and `question_encoder.embed_texts`
        `B * (1 + n_questions)` times per batch, one Python round-trip per
        example. Profiling (`in_dim=1024, d_model=256, B=16, T=150`, MPS)
        found the dominant cost was NOT the branch cross-attention itself
        (batching `self.branches` alone barely moved the needle in
        isolation) but the repeated `.to(device, dtype)` transfer inside
        `embed_texts` -- called on tiny tensors up to 64 times per forward
        -- which showed up as `aten::copy_`/`aten::to` at ~62% of self CPU
        time in `torch.profiler`. Collecting every text this call needs
        into ONE list and calling `embed_texts` once removes nearly all of
        that; batching `self.branches` per uniform-question-count group
        removes most of the remaining Python-loop and kernel-dispatch
        overhead.

        Two properties of the data this must respect (see `data.py`):
          - Augmentation (`permute_candidates`) is per-item, so two
            examples in the same batch can have a DIFFERENT NUMBER of
            options for the same question key. A naive `(B, Q, K, D)`
            reshape across the whole batch is wrong; only examples sharing
            the same option COUNT for a given key can be stacked together.
          - Examples can in principle carry different QUESTION KEYS (the
            `if label is None: continue` path in `ProsodiaDataset`), though
            MELD never exercises it (every example has all three labels).
            Examples are grouped by their exact ordered key tuple so a
            batched `self.branches` call only ever combines examples that
            asked the identical set of questions in the identical order.

        Within each of those constraints, this is mathematically the exact
        same computation as the reference loop: `nn.Linear` and
        `nn.MultiheadAttention` apply the same per-row op regardless of how
        many independent rows are processed in one call, so this and
        `_forward_looped_reference` should agree to float32 precision
        (see `tests/test_model.py::test_batched_forward_matches_looped_reference*`
        for the tolerance this was actually measured at).
        """
        h, mask = self._encode_state(batch)
        b = h.shape[0]
        results: list[dict[str, Tensor]] = [dict() for _ in range(b)]

        # ---- 1. One round trip for every text embedding this call needs.
        # `embed_texts` caches the frozen sentence-transformer encoding per
        # unique string, but each CALL still pays one device transfer and
        # one `nn.Linear` projection no matter how many strings it is
        # given -- so collecting everything into a single list turns
        # B*(1+Q) transfers/projections into one.
        all_texts: list[str] = []
        instr_spans: list[tuple[int, int]] = []
        opt_spans: list[dict[str, tuple[int, int]]] = []
        for questions in batch["questions"]:
            keys = list(questions)
            start = len(all_texts)
            all_texts.extend(questions[k]["instructions"] for k in keys)
            instr_spans.append((start, len(all_texts)))
            spans: dict[str, tuple[int, int]] = {}
            for k in keys:
                s = len(all_texts)
                all_texts.extend(questions[k]["options"])
                spans[k] = (s, len(all_texts))
            opt_spans.append(spans)

        if all_texts:
            all_vecs = self.question_encoder.embed_texts(all_texts)  # (N, D)
        else:
            # Every example in this batch had an empty question set --
            # nothing to embed. Shape/device/dtype must still match what
            # `embed_texts` would have returned so downstream indexing
            # (never reached below, since every group is skipped too) is
            # well-defined.
            all_vecs = h.new_zeros((0, self.d_model))

        # ---- 2. Group examples by their exact ordered key tuple so the
        # branch cross-attention runs as one batched call per group.
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, questions in enumerate(batch["questions"]):
            keys = tuple(questions)
            if not keys:
                continue
            groups.setdefault(keys, []).append(i)

        for keys, idxs in groups.items():
            q_vecs = torch.stack(
                [all_vecs[slice(*instr_spans[i])] for i in idxs], dim=0
            )                                                     # (Bg, Q, D)
            branch_out = self.branches(q_vecs, h[idxs], mask[idxs])  # (Bg, Q, D)

            for j, key in enumerate(keys):
                # ---- 3. Sub-batch the readout by option COUNT: ragged
                # per-item option counts mean only rows sharing the same K
                # can be stacked into one `self.readout` call.
                by_k: dict[int, list[int]] = {}
                for row, i in enumerate(idxs):
                    start, end = opt_spans[i][key]
                    by_k.setdefault(end - start, []).append(row)

                for rows in by_k.values():
                    opt_vecs = torch.stack(
                        [all_vecs[slice(*opt_spans[idxs[r]][key])] for r in rows], dim=0
                    )                                             # (m, K, D)
                    branch_vecs = branch_out[rows][:, j : j + 1]  # (m, 1, D)
                    logits = self.readout(branch_vecs, opt_vecs)  # (m, K)
                    for m_i, row in enumerate(rows):
                        results[idxs[row]][key] = logits[m_i]

        return results

    def _forward_looped_reference(self, batch: dict[str, Any]) -> list[dict[str, Tensor]]:
        """Pre-I6 implementation, kept ONLY as the equivalence oracle for
        `tests/test_model.py`'s batched-vs-looped tests. Not called by
        anything else -- this is the "current loop" the I6 fix is measured
        and checked against, deliberately preserved verbatim rather than
        re-derived, so the equivalence test cannot drift into comparing
        `forward` against a second, independently-written reimplementation
        of itself.
        """
        h, mask = self._encode_state(batch)
        results: list[dict[str, Tensor]] = []

        for i, questions in enumerate(batch["questions"]):
            keys = list(questions)
            if not keys:
                results.append({})
                continue

            q_vecs = self.question_encoder.embed_texts(
                [questions[k]["instructions"] for k in keys]
            ).unsqueeze(0)                                        # (1, Q, D)
            branch_out = self.branches(q_vecs, h[i : i + 1], mask[i : i + 1])

            per_question: dict[str, Tensor] = {}
            for j, key in enumerate(keys):
                opts = questions[key]["options"]
                opt_vecs = self.question_encoder.embed_texts(opts).unsqueeze(0)
                logits = self.readout(branch_out[:, j : j + 1], opt_vecs)
                per_question[key] = logits.squeeze(0)
            results.append(per_question)

        return results
