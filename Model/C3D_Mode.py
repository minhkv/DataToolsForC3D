from enum import Enum
class C3D_Mode(Enum):
    TRAINING = auto()
    TEST_TRAINED_NET = auto()
    FINE_TUNING = auto()
    TEST_FINE_TUNED_NET = auto()
    FEATURE_EXTRACTION = auto()
