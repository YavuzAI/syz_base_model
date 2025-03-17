import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications import InceptionV3 # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, 
    GlobalAveragePooling2D, Dense, Dropout, Add
) 
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

def create_model(input_shape=(512, 512, 4), num_classes=1):
    """
    Enhanced CNN model with better feature extraction and preprocessing 
    before feeding into InceptionV3.
    """

    # Input layer for 4-channel images
    inputs = Input(shape=input_shape)

    # **Preprocessing block to reduce 4 channels to 3 while preserving spatial features**
    x = Conv2D(32, (3, 3), padding='same')(inputs)  # , kernel_regularizer=l2(1e-4)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # **Residual Skip Connection**
    shortcut = Conv2D(64, (1, 1), padding="same")(inputs)  # 1x1 conv to match dimensions
    x = Add()([x, shortcut])  # Residual connection

    x = Conv2D(3, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # **MaxPooling before feeding into InceptionV3**
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Reducing size before passing to Inception

    # **Load pre-trained InceptionV3 (feature extractor)**
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(x.shape[0], x.shape[1], 3))
    base_model.trainable = False  # Initially freeze InceptionV3

    x = base_model(x)  # Pass through InceptionV3

    # **Global Average Pooling for feature extraction**
    x = GlobalAveragePooling2D()(x)

    # **Fully connected layers**
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # **Output layer for binary classification**
    outputs = Dense(num_classes, activation='sigmoid')(x)  # Binary classification

    model = Model(inputs, outputs)

    # **Compile with an adaptive learning rate scheduler**
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    return model
