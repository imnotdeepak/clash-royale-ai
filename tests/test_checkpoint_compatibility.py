"""Mandatory schema hashes; refuse loading across action schemas (MASTER_PLAN §6.4)."""

from __future__ import annotations

import dataclasses

import pytest

from royale_ai.adapters.action_schema import TILES_18X32_V1, SchemaMismatch, check_compatible


def test_same_schema_loads() -> None:
    check_compatible(TILES_18X32_V1.schema_id, TILES_18X32_V1.hash, TILES_18X32_V1)


def test_hash_is_stable_and_content_based() -> None:
    assert TILES_18X32_V1.hash == dataclasses.replace(TILES_18X32_V1).hash
    assert len(TILES_18X32_V1.hash) == 16


def test_v1_head_refused_as_v2() -> None:
    v2 = dataclasses.replace(
        TILES_18X32_V1, schema_id="tiles_18x32_v2", n_actions=2307, wait_actions=(0, 2305, 2306)
    )
    assert v2.hash != TILES_18X32_V1.hash
    with pytest.raises(SchemaMismatch):
        check_compatible(TILES_18X32_V1.schema_id, TILES_18X32_V1.hash, v2)
    with pytest.raises(SchemaMismatch):
        check_compatible(v2.schema_id, v2.hash, TILES_18X32_V1)


def test_same_id_different_content_refused() -> None:
    with pytest.raises(SchemaMismatch):
        check_compatible(TILES_18X32_V1.schema_id, "0" * 16, TILES_18X32_V1)
