from slicer.decorators import *


def test_norecurse():
    class DummyClass:
        @norecurse(default_return_value=100)
        def dummyFunc(self, value):
            return self.dummyFunc(value) + value

    assert DummyClass().dummyFunc(5) == 105

    class DummyClass:
        def __init__(self, default_value):
            self.default_value = default_value

        @norecurse(default_return_value=lambda self: self.default_value)
        def dummyFunc(self, value):
            return self.dummyFunc(value) + value

    assert DummyClass(20).dummyFunc(7) == 27
