"""Tests for src.constants."""

from __future__ import annotations


from src.constants import ESCI_LABEL2GAIN


def test_esci_label2gain_keys() -> None:
    """ESCI_LABEL2GAIN has exactly E, S, C, I."""
    assert set(ESCI_LABEL2GAIN.keys()) == {"E", "S", "C", "I"}


def test_esci_label2gain_values() -> None:
    """ESCI gains: E=1.0, S=0.1, C=0.01, I=0.0."""
    assert ESCI_LABEL2GAIN["E"] == 1.0
    assert ESCI_LABEL2GAIN["S"] == 0.1
    assert ESCI_LABEL2GAIN["C"] == 0.01
    assert ESCI_LABEL2GAIN["I"] == 0.0


def test_esci_label2gain_ordering() -> None:
    """E > S > C > I."""
    assert ESCI_LABEL2GAIN["E"] > ESCI_LABEL2GAIN["S"]
    assert ESCI_LABEL2GAIN["S"] > ESCI_LABEL2GAIN["C"]
    assert ESCI_LABEL2GAIN["C"] > ESCI_LABEL2GAIN["I"]
