from enum import Enum
class C3D_Mode(Enum):
    TRAINING = "conv3d_ucf101_train.prototxt"
    TEST_TRAINED_NET = "conv3d_ucf101_test.prototxt"
    FINE_TUNING = "c3d_ucf101_finetuning_train.prototxt"
    TEST_FINE_TUNED_NET = "c3d_ucf101_finetuning_test.prototxt"
    FEATURE_EXTRACTION_UCF101 = "c3d_ucf101_feature_extractor.prototxt"
    FEATURE_EXTRACTION_SPORT1M = "c3d_sport1m_feature_extractor.prototxt"
    DRAW_LEARNING_CURVE = "draw_learning_curve"
