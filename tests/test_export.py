"""GGUF export routing.

Conversion itself needs a llama.cpp clone and is exercised by hand, but the
decision this module makes -- which quantizer to use, and what to do when there
isn't one -- is pure logic and belongs under test. It is also where the real bug
was: the exporter used to raise when llama-quantize was missing, killing the run
*after* a valid f16 GGUF had been written and *before* the Modelfile, which is
the situation every Windows user without a C++ toolchain lands in.
"""

from __future__ import annotations

import pytest

from ftlab import export_gguf


@pytest.fixture
def merged(tmp_path):
    """A directory shaped like a merged model, enough to pass the guard."""
    path = tmp_path / "merged"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


@pytest.fixture
def fake_convert(monkeypatch):
    """Stand in for llama.cpp's converter, writing a plausible artefact."""
    def _convert(merged_dir, out_path, llama_cpp_dir=None, outtype="f16"):
        from pathlib import Path

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"GGUF" + b"\0" * 64)
        return out

    monkeypatch.setattr(export_gguf, "convert_to_gguf", _convert)


def _no_quantizer(monkeypatch):
    monkeypatch.setattr(export_gguf, "find_quantize_binary", lambda repo=None: None)
    monkeypatch.setattr(export_gguf, "_maybe_repo", lambda _d: None)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_missing_quantizer_returns_none_rather_than_raising(tmp_path):
    """Absence is a fallback, not an error -- the whole degradation path depends
    on this returning None."""
    assert export_gguf.find_quantize_binary(tmp_path) is None


def test_missing_llama_cpp_explains_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr(export_gguf.Path, "home", staticmethod(lambda: tmp_path))
    with pytest.raises(FileNotFoundError, match="git clone"):
        export_gguf.find_llama_cpp(None)


def test_pointing_at_an_adapter_is_caught_early(tmp_path):
    """A merged directory has config.json; an adapter does not.

    Checked before llama.cpp is located, so the message is the same whether or
    not a clone happens to exist on the machine.
    """
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(FileNotFoundError, match="Run 'ftlab merge' first"):
        export_gguf.convert_to_gguf(adapter, tmp_path / "x.gguf", tmp_path)


def test_unknown_quant_type_is_rejected(merged, tmp_path):
    with pytest.raises(ValueError, match="must be one of"):
        export_gguf.export(merged, tmp_path / "out", "m", quant="q9_bogus")


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def test_no_quantizer_keeps_f16_and_still_writes_the_modelfile(
    merged, tmp_path, monkeypatch, fake_convert
):
    """The regression that motivated the rewrite: a missing compiler must
    downgrade the result, not destroy it."""
    _no_quantizer(monkeypatch)
    monkeypatch.setattr(export_gguf, "ollama_available", lambda: False)

    result = export_gguf.export(merged, tmp_path / "out", "demo", quant="q4_k_m")

    assert result.quant_route == "none"
    assert result.final_path == result.f16_path
    assert result.final_path.exists()
    assert result.modelfile.exists()
    assert "shipping f16" in result.summary()


def test_ollama_quantizes_on_ingest_when_no_binary_exists(
    merged, tmp_path, monkeypatch, fake_convert
):
    """The route that matters on Windows: no compiler needed."""
    _no_quantizer(monkeypatch)
    monkeypatch.setattr(export_gguf, "ollama_available", lambda: True)
    calls = {}
    monkeypatch.setattr(
        export_gguf,
        "register_with_ollama",
        lambda mf, name, quantize_to=None: calls.update(name=name, quant=quantize_to),
    )

    result = export_gguf.export(
        merged, tmp_path / "out", "demo", quant="q4_k_m", ollama_name="demo-model"
    )

    assert result.quant_route == "ollama"
    # The Modelfile still points at the f16; ollama does the work on ingest.
    assert result.final_path == result.f16_path
    assert calls == {"name": "demo-model", "quant": "q4_k_m"}


def test_a_present_binary_wins_and_produces_a_portable_file(
    merged, tmp_path, monkeypatch, fake_convert
):
    monkeypatch.setattr(export_gguf, "_maybe_repo", lambda _d: tmp_path)
    monkeypatch.setattr(
        export_gguf, "find_quantize_binary", lambda repo=None: tmp_path / "llama-quantize"
    )

    def _quantize(src, out, quant_type="q4_k_m", llama_cpp_dir=None):
        from pathlib import Path

        Path(out).write_bytes(b"GGUF")
        return Path(out)

    monkeypatch.setattr(export_gguf, "quantize", _quantize)

    result = export_gguf.export(merged, tmp_path / "out", "demo", quant="q4_k_m")

    assert result.quant_route == "llama-quantize"
    assert result.final_path != result.f16_path
    assert result.final_path.name == "demo-q4_k_m.gguf"


def test_f16_target_skips_quantization_entirely(
    merged, tmp_path, monkeypatch, fake_convert
):
    _no_quantizer(monkeypatch)
    result = export_gguf.export(merged, tmp_path / "out", "demo", quant="f16")
    assert result.quant_route == "none"
    assert result.final_path == result.f16_path


# ---------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------


def test_modelfile_points_at_the_gguf_and_carries_the_system_prompt(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    path = export_gguf.write_modelfile(
        gguf, tmp_path / "Modelfile", system_prompt='Answer from the "library".'
    )
    text = path.read_text(encoding="utf-8")

    assert f"FROM {gguf.resolve()}" in text
    assert "PARAMETER temperature" in text
    # Quotes must be escaped or the triple-quoted SYSTEM block terminates early.
    assert 'Answer from the \\"library\\".' in text


def test_modelfile_omits_the_system_block_when_there_is_no_prompt(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    text = export_gguf.write_modelfile(gguf, tmp_path / "Modelfile").read_text(
        encoding="utf-8"
    )
    assert "SYSTEM" not in text


def test_quant_names_are_translated_for_ollama():
    """ollama spells these in caps; passing the lowercase form is rejected."""
    assert export_gguf.OLLAMA_QUANT["q4_k_m"] == "q4_K_M"
    assert export_gguf.OLLAMA_QUANT["q6_k"] == "q6_K"
    assert set(export_gguf.OLLAMA_QUANT) <= set(export_gguf.QUANT_TYPES)
