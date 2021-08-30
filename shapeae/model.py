from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, BatchNormalization, \
    Conv3DTranspose, Dense, GlobalAveragePooling3D, AveragePooling3D

from tensorflow.keras.models import Model
from .metrics import *
from tensorflow.keras.layers import Dropout as DP
from tensorflow.keras.optimizers import Adam


def model(params):
    num_filters = params["num_filters"]
    dropout = params["dropout"]
    print(num_filters, dropout)
    """
    Encoder model for encoding the 2D images in three convolutional blocks
    Input: The encoder expext the input data in the format (1, x-dim, y-dim, 2), while the last dimension are the two channels
    Output: It will return tensor of shape (1, x-dim/3, y-dim/3, 1)
    """

    mask_input_img = Input(shape=(1, None, None, 2))  # adapt this if using `channels_first` image data format
    mx = Conv3D(num_filters, (1, 3, 3), activation='relu', padding='same')(mask_input_img)
    mx = BatchNormalization()(mx)
    mx = Conv3D(num_filters, (1, 3, 3), activation='relu', padding='same')(mx)
    mx = BatchNormalization()(mx)
    mx = DP(dropout)(mx)

    mx = MaxPooling3D((1, 2, 2), padding='same')(mx)
    mx = Conv3D(num_filters * 4, (1, 3, 3), activation='relu', padding='same')(mx)
    mx = BatchNormalization()(mx)
    mx = Conv3D(num_filters * 4, (1, 3, 3), activation='relu', padding='same')(mx)
    mx = BatchNormalization()(mx)
    mx = DP(dropout)(mx)

    mx = MaxPooling3D((1, 2, 2), padding='same')(mx)
    mx = Conv3D(num_filters * 8, (1, 3, 3), activation='relu', padding='same')(mx)
    mx = BatchNormalization()(mx)
    mx = Conv3D(num_filters * 8, (1, 3, 3), activation='relu', padding='same')(mx)
    mx = BatchNormalization()(mx)
    mx = DP(dropout)(mx)

    mask_encoded = Conv3D(1, (1, 3, 3), activation='sigmoid', padding='same')(mx)
    mask_encoder = Model(inputs=mask_input_img, outputs=mask_encoded, name="Maskencoder")
    mask_encoder.summary()

    """
    Decoder model for decoding the 2D features from the encoder to a 2D shape with the same dimesions as the 2D image, but 64 z-dimensions
    Input: The decoder expexts the input data in the format (1, x-dim, y-dim, 2), while the last dimension are the two channels
    Output: It will return tensor of shape (64, x-dim*3, y-dim*3, 1)
    """
    decoder_input_img = Input(shape=(1, None, None, 1))
    x = Conv3D(num_filters * 8, (1, 3, 3), activation='relu', padding='same')(decoder_input_img)
    x = BatchNormalization()(x)
    x = Conv3D(num_filters * 8, (1, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = Conv3DTranspose(num_filters * 8, (3, 3, 3), strides=(2, 1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(num_filters * 8, (2, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = Conv3DTranspose(num_filters * 4, (2, 3, 3), strides=(2, 1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(num_filters * 4, (2, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = Conv3DTranspose(num_filters * 4, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(num_filters * 4, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = Conv3DTranspose(num_filters * 2, (3, 3, 3), strides=(2, 1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(num_filters * 2, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = Conv3DTranspose(num_filters, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = Conv3DTranspose(num_filters, (2, 3, 3), strides=(2, 1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    decoded = Conv3D(1, (2, 3, 3), activation='sigmoid', padding='same', name="decoder_output")(x)
    decoder = Model(decoder_input_img, decoded, name="Decoder")
    decoder.summary()

    ShapeAEmodel = Model(inputs=mask_input_img, outputs=decoder(mask_encoder(mask_input_img)))
    ShapeAEmodel.compile(optimizer='adam', loss=dice_crossentropy_loss, metrics=["mse"])
    ShapeAEmodel.summary()

    discriminator_input_img = Input(shape=(None, None, None, 1))
    x = Conv3D(num_filters, (3, 3, 3), activation='relu', padding='same')(discriminator_input_img)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = AveragePooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(num_filters*2, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = AveragePooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(num_filters*4, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = AveragePooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(num_filters*8, (3,3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = AveragePooling3D((2, 2, 2), padding='same')(x)
    x = Conv3D(num_filters*16, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = DP(dropout)(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=256, activation="relu")(x)
    x = DP(dropout)(x)
    x = Dense(1, activation="sigmoid")(x)

    discriminator = Model(inputs=discriminator_input_img, outputs=x)
    discriminator.compile(optimizer=Adam(lr=0.00000005, beta_1=0.9), loss="binary_crossentropy", metrics=["accuracy"])
    discriminator.summary()

    return ShapeAEmodel, discriminator


def define_adverserial(ShapeAEmodel, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    input_image = Input(shape=(1, 64, 64, 2))
    adverserialmodel = Model(inputs=input_image,
                             outputs=[ShapeAEmodel(input_image), discriminator(ShapeAEmodel(input_image))])
    adverserialmodel.compile(loss=[dice_crossentropy_loss, "binary_crossentropy"], loss_weights=[10, 1],
                             optimizer="adam")
    return adverserialmodel