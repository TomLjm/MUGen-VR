from mugen.generation.generators.anyflow_generator import build_chunk_partition


def test_chunk_partition_preserves_upstream_default():
    default = [1, 3, 3, 3, 3, 3, 3, 2]
    assert build_chunk_partition(81, 4, default) == default


def test_chunk_partition_adapts_25_and_49_frame_runs():
    default = [1, 3, 3, 3, 3, 3, 3, 2]
    assert build_chunk_partition(25, 4, default) == [1, 3, 3]
    assert build_chunk_partition(49, 4, default) == [1, 3, 3, 3, 3]
