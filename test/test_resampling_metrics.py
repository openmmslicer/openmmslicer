from slicer.resampling_metrics import *


degenerate_weights = [1, 0, 0, 0]
diverse_weights = [0.10, 0.15, 0.35, 0.4]
equal_weights = [0.25, 0.25, 0.25, 0.25]


def test_EffectiveSampleSize():
    assert EffectiveSampleSize.evaluate(degenerate_weights) == 0.25
    assert EffectiveSampleSize.evaluate(diverse_weights) == 0.7936507996578126
    assert EffectiveSampleSize.evaluate(equal_weights) == 1


def test_ExpectedSampleSize():
    assert ExpectedSampleSize.evaluate(degenerate_weights) == 0.25
    assert ExpectedSampleSize.evaluate(diverse_weights) == 0.75
    assert ExpectedSampleSize.evaluate(equal_weights) == 1


def test_WorstCaseSampleSize():
    assert WorstCaseSampleSize.evaluate(degenerate_weights) == 0.25
    assert WorstCaseSampleSize.evaluate(diverse_weights) == 0.625
    assert WorstCaseSampleSize.evaluate(equal_weights) == 1


def test_WorstCaseSystematicSampleSize():
    assert WorstCaseSystematicSampleSize.evaluate(degenerate_weights) == 0.25
    assert WorstCaseSystematicSampleSize.evaluate(diverse_weights) == 0.5
    assert WorstCaseSystematicSampleSize.evaluate(equal_weights) == 1


def test_ExpWeightEntropy():
    assert ExpWeightEntropy.evaluate(degenerate_weights) == 0.25
    assert ExpWeightEntropy.evaluate(diverse_weights) == 0.8715221881866455
    assert ExpWeightEntropy.evaluate(equal_weights) == 1
