from tensorflow import keras
from tensorflow.keras import layers, Model

from Attention import MultiDepthFusion, MultiModalFusion
from BasicModule import Classifier, EEG_Head, GSR_Head
from Distill_zoo import MLKDLoss, feedback_Loss
from Transformer_model import transformer_feature_extractor, PositionEmbedding, TransformerBlock, \
    transformer_feature_extracting


def get_GSR_model(KD_mode="MLKD", learning_rate=1e-6, e2=0.1, e3=1):
    input_ori_gsr = layers.Input((1, 512, 1), name="GSR_input")
    input_hard_label = layers.Input(2, name="hard_label")
    input_soft_label = layers.Input(2, name="soft_label")

    gsr_feature = GSR_Head()(input_ori_gsr)
    gsr_feature_1, gsr_feature_2, gsr_feature_3 = transformer_feature_extractor()(gsr_feature)
    att_feature = MultiDepthFusion(64, 64)([gsr_feature_1, gsr_feature_2, gsr_feature_3])
    att_feature = layers.Flatten(name="att_feature")(att_feature)

    logit_feature = Classifier(hide_dim=128, output_dim=2)(att_feature)
    cls = layers.Activation(activation="softmax", name="cls")(logit_feature)

    input_soft_feature = layers.Input(shape=(3904,), name="soft_feature")

    if KD_mode == "MLKD":
        print("Training with MLKD")
        cls = MLKDLoss(e2=e2, e3=e3)([input_hard_label, input_soft_label, input_soft_feature, att_feature, cls])

        model = Model([input_ori_gsr, input_hard_label, input_soft_label, input_soft_feature], cls)
        model_opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=model_opt)
    else:
        model = Model(input_ori_gsr, [cls, logit_feature, att_feature], name="gsr_student")
        model_opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=model_opt, loss="categorical_crossentropy", metrics=["acc"])

    return model

def get_MultiModal_model(learning_rate=5e-5):
    input_ori_gsr = layers.Input((1, 512, 1))
    input_ori_eeg = layers.Input((28, 512, 1))

    gsr_feature = GSR_Head()(input_ori_gsr)
    eeg_feature = EEG_Head()(input_ori_eeg)

    eeg_feature_1, eeg_feature_2, eeg_feature_3 = transformer_feature_extracting(eeg_feature)
    gsr_feature_1, gsr_feature_2, gsr_feature_3 = transformer_feature_extracting(gsr_feature)

    att_feature_1 = MultiModalFusion(64, 64)([eeg_feature_1, gsr_feature_1, eeg_feature_2, gsr_feature_2])
    att_feature_2 = MultiModalFusion(64, 64)([eeg_feature_2, gsr_feature_2, eeg_feature_3, gsr_feature_3])
    att_feature = att_feature_1 + att_feature_2

    att_feature = layers.Flatten(name="att_feature")(att_feature)

    # Ensure the shape of att_feature matches the expected shape in GSR model
    att_feature = layers.Dense(3904, activation='relu')(att_feature)

    logit_feature = Classifier(hide_dim=128, output_dim=2)(att_feature)
    cls = layers.Activation(activation="softmax", name="prediction")(logit_feature)

    model = Model([input_ori_eeg, input_ori_gsr], [cls, logit_feature, att_feature])
    model.summary()
    model_opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=model_opt, loss=["categorical_crossentropy", None, None], metrics=["acc"])

    return model


if __name__ == "__main__":
    get_MultiModal_model()
    get_GSR_model()