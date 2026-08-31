"""The loss mask is the load-bearing part of QRA training, so it gets asserted
character by character rather than by shape alone."""

from __future__ import annotations

import pytest

from ftlab.shared.data import IGNORE_INDEX, QRAExample, encode, encode_all, render_assistant

EXAMPLE = QRAExample(
    question="What is 2+2?",
    reasoning="Two plus two is four by addition.",
    answer="4",
)


def scored_text(row, tokenizer) -> str:
    return tokenizer.decode(
        [t for t, label in zip(row["input_ids"], row["labels"], strict=True)
         if label != IGNORE_INDEX]
    )


def masked_text(row, tokenizer) -> str:
    return tokenizer.decode(
        [t for t, label in zip(row["input_ids"], row["labels"], strict=True)
         if label == IGNORE_INDEX]
    )


def test_lengths_line_up(tokenizer, data_cfg):
    row = encode(EXAMPLE, data_cfg, tokenizer)
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])


def test_labels_mirror_inputs_where_not_masked(tokenizer, data_cfg):
    """Where a label is scored it must equal its own input token, not a shift.

    The causal shift is the model's job; doing it here too would train the model
    to predict the token it was just given.
    """
    row = encode(EXAMPLE, data_cfg, tokenizer)
    for token, label in zip(row["input_ids"], row["labels"], strict=True):
        assert label in (IGNORE_INDEX, token)


def test_question_is_never_scored(tokenizer, data_cfg):
    row = encode(EXAMPLE, data_cfg, tokenizer)
    assert EXAMPLE.question not in scored_text(row, tokenizer)
    assert EXAMPLE.question in masked_text(row, tokenizer)


def test_reasoning_and_answer_both_scored_by_default(tokenizer, data_cfg):
    data_cfg.train_on_reasoning = True
    scored = scored_text(encode(EXAMPLE, data_cfg, tokenizer), tokenizer)
    assert EXAMPLE.reasoning in scored
    assert EXAMPLE.answer in scored


def test_reasoning_masked_when_disabled(tokenizer, data_cfg):
    data_cfg.train_on_reasoning = False
    row = encode(EXAMPLE, data_cfg, tokenizer)

    scored, masked = scored_text(row, tokenizer), masked_text(row, tokenizer)
    assert EXAMPLE.reasoning not in scored, "trace leaked into the scored span"
    assert EXAMPLE.reasoning in masked, "trace should still be in context"
    assert EXAMPLE.answer in scored


def test_no_duplicate_bos(tokenizer, data_cfg):
    """The template emits BOS; the tokenizer must not add a second one."""
    from tests.conftest import BOS

    row = encode(EXAMPLE, data_cfg, tokenizer)
    assert row["input_ids"].count(ord(BOS)) == 1


def test_system_prompt_is_masked(tokenizer, data_cfg):
    data_cfg.system_prompt = "You are a careful reasoner."
    row = encode(EXAMPLE, data_cfg, tokenizer)
    assert data_cfg.system_prompt in masked_text(row, tokenizer)
    assert data_cfg.system_prompt not in scored_text(row, tokenizer)


def test_answer_only_format_drops_the_trace(tokenizer, data_cfg):
    data_cfg.reasoning_format = "answer_only"
    row = encode(EXAMPLE, data_cfg, tokenizer)
    whole = tokenizer.decode(row["input_ids"])
    assert EXAMPLE.reasoning not in whole
    assert EXAMPLE.answer in scored_text(row, tokenizer)


def test_labeled_format_uses_plain_headings(tokenizer, data_cfg):
    data_cfg.reasoning_format = "labeled"
    text, offset = render_assistant(EXAMPLE, data_cfg)
    assert text.startswith("Reasoning:\n")
    assert text[offset:] == EXAMPLE.answer


def test_think_tags_offset_points_at_the_answer(tokenizer, data_cfg):
    text, offset = render_assistant(EXAMPLE, data_cfg)
    assert text.startswith("<think>")
    assert text[offset:] == EXAMPLE.answer


def test_missing_reasoning_falls_back_to_answer_only(tokenizer, data_cfg):
    example = QRAExample(question="Q", answer="A", reasoning="")
    text, offset = render_assistant(example, data_cfg)
    assert text == "A"
    assert offset == 0


def test_non_prefix_template_is_rejected(data_cfg):
    """A template we cannot mask must raise, not train on a wrong mask."""
    from tests.conftest import NonPrefixTokenizer

    with pytest.raises(ValueError, match="prefix-stable"):
        encode(EXAMPLE, data_cfg, NonPrefixTokenizer())


def test_overlong_examples_are_dropped(tokenizer, data_cfg):
    data_cfg.max_seq_len = 40  # keeps the 24-token example, drops the 87-token one
    data_cfg.drop_overlong = True
    rows, stats = encode_all([EXAMPLE, QRAExample(question="a", answer="b")], data_cfg, tokenizer)
    assert stats["dropped"] == 1
    assert stats["kept"] == 1
    assert all(len(row["input_ids"]) <= 40 for row in rows)


def test_everything_dropped_is_an_error(tokenizer, data_cfg):
    data_cfg.max_seq_len = 2
    data_cfg.drop_overlong = True
    with pytest.raises(ValueError, match="every example was dropped"):
        encode_all([EXAMPLE], data_cfg, tokenizer)


def test_truncation_keeps_all_three_fields_aligned(tokenizer, data_cfg):
    data_cfg.max_seq_len = 50  # past the 32-token prompt, so answer tokens survive
    data_cfg.drop_overlong = False
    rows, stats = encode_all([EXAMPLE], data_cfg, tokenizer)
    assert stats["truncated"] == 1
    row = rows[0]
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"]) == 50


def test_truncation_that_erases_every_scored_token_drops_the_row(tokenizer, data_cfg):
    """Truncating back into the prompt leaves an all-masked row.

    Such a row contributes no gradient but would still be counted in the batch,
    quietly shrinking the effective batch size, so it must be dropped instead.
    """
    data_cfg.max_seq_len = 30  # inside the 32-token prompt
    data_cfg.drop_overlong = False
    with pytest.raises(ValueError, match="every example was dropped"):
        encode_all([EXAMPLE], data_cfg, tokenizer)


# ---------------------------------------------------------------------------
# reasoning models whose template owns the thinking span
# ---------------------------------------------------------------------------


def test_think_tags_are_rejected_on_a_thinking_template(gemma_tokenizer, data_cfg):
    """The real Gemma 4 failure, reproduced.

    Its generation prompt opens and closes an empty thought channel, so a
    <think>-in-content rendering is not a prefix of the full conversation.
    Training on a guessed boundary would be worse than refusing.
    """
    data_cfg.reasoning_format = "think_tags"
    with pytest.raises(ValueError, match="prefix-stable"):
        encode(EXAMPLE, data_cfg, gemma_tokenizer)


def test_the_error_points_at_the_remedy(gemma_tokenizer, data_cfg):
    """A refusal with no way forward is only half a diagnosis."""
    data_cfg.reasoning_format = "think_tags"
    with pytest.raises(ValueError, match="native"):
        encode(EXAMPLE, data_cfg, gemma_tokenizer)
    with pytest.raises(ValueError, match="enable_thinking"):
        encode(EXAMPLE, data_cfg, gemma_tokenizer)


def test_native_rendering_with_thinking_enabled_masks_correctly(gemma_tokenizer, data_cfg):
    data_cfg.reasoning_format = "native"
    data_cfg.chat_template_kwargs = {"enable_thinking": True}
    data_cfg.system_prompt = "SYSTEM"

    row = encode(EXAMPLE, data_cfg, gemma_tokenizer)
    scored, masked = scored_text(row, gemma_tokenizer), masked_text(row, gemma_tokenizer)

    assert EXAMPLE.question in masked
    assert data_cfg.system_prompt in masked
    # The template placed the trace in its own channel; both it and the answer
    # are the model's job to produce.
    assert EXAMPLE.reasoning in scored
    assert EXAMPLE.answer in scored
    assert "<|channel>thought" in scored


def test_native_can_still_mask_the_trace_alone(gemma_tokenizer, data_cfg):
    """train_on_reasoning=False must work even when the template owns the span.

    The offset is recovered from the rendered completion rather than from our
    own string, since the template decides where the trace goes.
    """
    data_cfg.reasoning_format = "native"
    data_cfg.chat_template_kwargs = {"enable_thinking": True}
    data_cfg.train_on_reasoning = False

    row = encode(EXAMPLE, data_cfg, gemma_tokenizer)
    scored, masked = scored_text(row, gemma_tokenizer), masked_text(row, gemma_tokenizer)

    assert EXAMPLE.reasoning not in scored
    assert EXAMPLE.reasoning in masked
    assert EXAMPLE.answer in scored


def test_template_kwargs_reach_the_prompt_renderer(gemma_tokenizer, data_cfg):
    """Inference must render the prompt exactly as training did.

    Passing the kwargs on one side only would give the model a format at
    inference that it never saw in training.
    """
    from ftlab.shared.data import render_prompt

    data_cfg.chat_template_kwargs = {"enable_thinking": True}
    prompt = render_prompt(EXAMPLE, data_cfg, gemma_tokenizer)
    assert "<|think|>" in prompt
    assert "<|channel>thought\n<channel|>" not in prompt


def test_no_reasoning_under_native_degrades_to_a_plain_answer(gemma_tokenizer, data_cfg):
    data_cfg.reasoning_format = "native"
    data_cfg.chat_template_kwargs = {"enable_thinking": True}
    example = QRAExample(question="Q", answer="A", reasoning="")

    row = encode(example, data_cfg, gemma_tokenizer)
    assert "<|channel>thought" not in tokenizer_decode(row, gemma_tokenizer)
    assert "A" in scored_text(row, gemma_tokenizer)


def tokenizer_decode(row, tokenizer):
    return tokenizer.decode(row["input_ids"])
