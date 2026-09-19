# tests/test_qencoder.py
import torch

from prosodia.model.qencoder import QuestionEncoder


def test_embeddings_are_projected_to_d_model():
    enc = QuestionEncoder(d_model=64)
    out = enc.embed_texts(["Which emotion is the speaker expressing?",
                           "How does the speaker feel here?"])
    assert out.shape == (2, 64)


def test_paraphrases_are_closer_than_unrelated_questions():
    enc = QuestionEncoder(d_model=64)
    v = enc.embed_texts([
        "Which emotion is the speaker expressing?",
        "What emotion comes through in this utterance?",
        "What is the account balance for this customer?",
    ])
    v = torch.nn.functional.normalize(v, dim=-1)
    assert (v[0] @ v[1]) > (v[0] @ v[2])


def test_repeated_text_is_cached_not_recomputed():
    enc = QuestionEncoder(d_model=64)
    enc.embed_texts(["Which emotion is the speaker expressing?"])
    before = enc.cache_misses
    enc.embed_texts(["Which emotion is the speaker expressing?"])
    assert enc.cache_misses == before


def test_only_projection_is_trainable():
    enc = QuestionEncoder(d_model=64)
    assert all(not p.requires_grad for p in enc._st.parameters())
    assert all(p.requires_grad for p in enc.project.parameters())


def test_sentence_transformer_stays_in_eval_mode_when_parent_trains():
    # A future training loop will call .train() on a larger model this
    # encoder is nested in. The frozen sentence-transformer must not
    # flip into train mode (which would enable dropout and make its
    # "frozen" embeddings nondeterministic).
    enc = QuestionEncoder(d_model=64)
    enc.train()
    assert enc._st.training is False
    assert enc.project.training is True
