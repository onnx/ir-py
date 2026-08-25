# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared shard filename generation."""

from __future__ import annotations

import os


def _is_extension_suffix(suffix: str) -> bool:
    extension = suffix[1:]
    return (
        bool(extension)
        and extension[0].isascii()
        and extension[0].isalpha()
        and all(
            character.isascii() and (character.isalnum() or character == "_")
            for character in extension
        )
    )


def get_shard_filename(
    base_name: str,
    shard_idx: int,
    total_shards: int,
    *,
    suffix_count: int | None = None,
) -> str:
    """Insert a shard marker before a filename's extension suffix chain."""
    if total_shards == 1:
        return base_name

    dir_name, filename = os.path.split(base_name)
    name = filename
    suffixes: list[str] = []
    while suffix_count is None or len(suffixes) < suffix_count:
        candidate_name, suffix = os.path.splitext(name)
        if not suffix or not _is_extension_suffix(suffix):
            break
        name = candidate_name
        suffixes.append(suffix)
    extension = "".join(reversed(suffixes))

    shard_filename = f"{name}-{shard_idx:05d}-of-{total_shards:05d}{extension}"
    return os.path.join(dir_name, shard_filename) if dir_name else shard_filename
