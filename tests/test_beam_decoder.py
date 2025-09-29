import numpy as np
import pytest
from src.metrics.beam_decoder import BeamCTCDecoder


def test_simple_sequence():
    """Проверяем что beam search выбирает последовательность с самой большой вероятностью"""
    # 3 шага, 3 символа (0=blank, 1, 2)
    log_probs = np.log(np.array([
        [0.7, 0.2, 0.1],  # шаг 1 → blank
        [0.1, 0.8, 0.1],  # шаг 2 → "1"
        [0.1, 0.2, 0.7],  # шаг 3 → "2"
    ]))

    decoder = BeamCTCDecoder(blank=0, beam_width=3)
    result = decoder.decode(log_probs)

    assert result == [1, 2], f"Ожидали [1, 2], получили {result}"


def test_repeated_symbols_are_collapsed():
    """Повторяющиеся подряд символы должны схлопываться"""
    log_probs = np.log(np.array([
        [0.1, 0.9],  # "1"
        [0.1, 0.9],  # "1" повтор
        [0.9, 0.1],  # blank
        [0.1, 0.9],  # снова "1"
    ]))

    decoder = BeamCTCDecoder(blank=0, beam_width=2)
    result = decoder.decode(log_probs)

    assert result == [1, 1], f"Ожидали [1, 1], получили {result}"


def test_blank_is_removed():
    """blank символы должны убираться"""
    log_probs = np.log(np.array([
        [0.9, 0.1],  # blank
        [0.1, 0.9],  # "1"
        [0.9, 0.1],  # blank
        [0.1, 0.9],  # "1"
    ]))

    decoder = BeamCTCDecoder(blank=0, beam_width=2)
    result = decoder.decode(log_probs)

    assert result == [1, 1], f"Ожидали [1, 1], получили {result}"


def test_argmax_equivalence_on_easy_case():
    """Beam search должен совпадать с argmax в простом случае"""
    log_probs = np.log(np.array([
        [0.05, 0.9, 0.05],  # "1"
        [0.05, 0.05, 0.9],  # "2"
    ]))

    argmax_result = [1, 2]

    decoder = BeamCTCDecoder(blank=0, beam_width=3)
    result = decoder.decode(log_probs)

    assert result == argmax_result, f"Ожидали {argmax_result}, получили {result}"
