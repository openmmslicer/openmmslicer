from slicer.resampling_metrics import *


degenerate_weights = [1, 0, 0, 0]
diverse_weights = [0.10, 0.15, 0.35, 0.4]
equal_weights = [0.25, 0.25, 0.25, 0.25]


def test_EffectiveSampleSize():
    assert EffectiveSampleSize()(degenerate_weights) == 0.25
    assert EffectiveSampleSize()(diverse_weights) == 0.7936507996578126
    assert EffectiveSampleSize()(equal_weights) == 1


def test_ExpectedSampleSize():
    assert ExpectedSampleSize()(degenerate_weights) == 0.25
    assert ExpectedSampleSize()(diverse_weights) == 0.75
    assert ExpectedSampleSize()(equal_weights) == 1


def test_WorstCaseSampleSize():
    assert WorstCaseSampleSize()(degenerate_weights) == 0.25
    assert WorstCaseSampleSize()(diverse_weights) == 0.625
    assert WorstCaseSampleSize()(equal_weights) == 1


def test_WorstCaseSystematicSampleSize():
    assert WorstCaseSystematicSampleSize()(degenerate_weights) == 0.25
    assert WorstCaseSystematicSampleSize()(diverse_weights) == 0.5
    assert WorstCaseSystematicSampleSize()(equal_weights) == 1


def test_ExpWeightEntropy():
    assert ExpWeightEntropy()(degenerate_weights) == 0.25
    assert ExpWeightEntropy()(diverse_weights) == 0.8715221881866455
    assert ExpWeightEntropy()(equal_weights) == 1
