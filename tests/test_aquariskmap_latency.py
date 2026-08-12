from scripts.benchmark_aquariskmap_latency import MEASURE_IMAGES, WARMUP_IMAGES


def test_latency_protocol_uses_preregistered_warmup_and_measurement_counts():
    assert WARMUP_IMAGES == 50
    assert MEASURE_IMAGES == 50
