# src/model_def.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    GlobalAveragePooling2D, Dense, Reshape, Add, Activation, BatchNormalization, Average
)
from tensorflow.keras.models import Model

def build_example_encoder(input_shape=(64, 64, 1), latent_dim=64, name="example_encoder"):
    # Input: 128x128x1
    img_input = Input(shape=input_shape, name=f"{name}_input")
    x = Conv2D(16, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv1")(img_input) # 64x64
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Activation("relu", name=f"{name}_relu1")(x)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv2")(x) # 32x32
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = Activation("relu", name=f"{name}_relu2")(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv3")(x) # 16x16
    x = BatchNormalization(name=f"{name}_bn3")(x)
    x = Activation("relu", name=f"{name}_relu3")(x)
    
    x = Conv2D(latent_dim, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv4")(x) # 8x8
    x = BatchNormalization(name=f"{name}_bn4")(x)
    x = Activation("relu", name=f"{name}_relu4")(x)
    
    feature_vector = GlobalAveragePooling2D(name=f"{name}_gap")(x)
    encoder = Model(img_input, feature_vector, name=name)
    return encoder

def build_generator(outline_input_shape=(64, 64, 1), example_latent_dim=64, num_filters_base=16, name="generator"):
    outline_input = Input(shape=outline_input_shape, name=f"{name}_outline_input")
    condition_input = Input(shape=(example_latent_dim,), name=f"{name}_condition_input")

    # Encoder Path
    e1 = Conv2D(num_filters_base, (3,3), padding="same", name=f"{name}_e1_conv")(outline_input)
    e1 = BatchNormalization(name=f"{name}_e1_bn")(e1); e1 = Activation("relu", name=f"{name}_e1_relu")(e1) # 128x128
    p1 = MaxPooling2D((2,2), name=f"{name}_e1_pool")(e1) # 64x64

    e2 = Conv2D(num_filters_base*2, (3,3), padding="same", name=f"{name}_e2_conv")(p1)
    e2 = BatchNormalization(name=f"{name}_e2_bn")(e2); e2 = Activation("relu", name=f"{name}_e2_relu")(e2) # 64x64
    p2 = MaxPooling2D((2,2), name=f"{name}_e2_pool")(e2) # 32x32

    e3 = Conv2D(num_filters_base*4, (3,3), padding="same", name=f"{name}_e3_conv")(p2)
    e3 = BatchNormalization(name=f"{name}_e3_bn")(e3); e3 = Activation("relu", name=f"{name}_e3_relu")(e3) # 32x32
    p3 = MaxPooling2D((2,2), name=f"{name}_e3_pool")(e3) # 16x16

    e4 = Conv2D(num_filters_base*8, (3,3), padding="same", name=f"{name}_e4_conv")(p3)
    e4 = BatchNormalization(name=f"{name}_e4_bn")(e4); e4 = Activation("relu", name=f"{name}_e4_relu")(e4) # 16x16
    p4 = MaxPooling2D((2,2), name=f"{name}_e4_pool")(e4) # 8x8

    # Bottleneck
    b = Conv2D(num_filters_base*16, (3,3), padding="same", name=f"{name}_b_conv")(p4) # 8x8
    b = BatchNormalization(name=f"{name}_b_bn")(b); b = Activation("relu", name=f"{name}_b_relu")(b) # 8x8x(NFB*16)

    # Conditioning: Project condition_input and add to bottleneck
    # Target spatial dims for condition: b.shape[1:3] (e.g., (8,8))
    # Target channels for condition: b.shape[-1] (e.g., num_filters_base*8)
    cond_channels = b.shape[-1]
    cond_spatial_dims = b.shape[1:3] # (8,8)
    
    projected_condition = Dense(cond_spatial_dims[0] * cond_spatial_dims[1] * cond_channels, activation='relu', name=f"{name}_cond_dense")(condition_input)
    reshaped_condition = Reshape((cond_spatial_dims[0], cond_spatial_dims[1], cond_channels), name=f"{name}_cond_reshape")(projected_condition)
    b_conditioned = Add(name=f"{name}_cond_add")([b, reshaped_condition])

    # Decoder Path
    d1 = UpSampling2D((2,2), name=f"{name}_d1_upsample")(b_conditioned) # From 8x8 to 16x16
    d1 = concatenate([d1, e4], name=f"{name}_d1_concat") # Skip from e4 (16x16)
    d1 = Conv2D(num_filters_base*8, (3,3), padding="same", name=f"{name}_d1_conv")(d1)
    d1 = BatchNormalization(name=f"{name}_d1_bn")(d1); d1 = Activation("relu", name=f"{name}_d1_relu")(d1)

    d2 = UpSampling2D((2,2), name=f"{name}_d2_upsample")(d1) # From 16x16 to 32x32
    d2 = concatenate([d2, e3], name=f"{name}_d2_concat") # Skip from e3 (32x32)
    d2 = Conv2D(num_filters_base*4, (3,3), padding="same", name=f"{name}_d2_conv")(d2)
    d2 = BatchNormalization(name=f"{name}_d2_bn")(d2); d2 = Activation("relu", name=f"{name}_d2_relu")(d2)

    d3 = UpSampling2D((2,2), name=f"{name}_d3_upsample")(d2) # From 32x32 to 64x64
    d3 = concatenate([d3, e2], name=f"{name}_d3_concat") # Skip from e2 (64x64)
    d3 = Conv2D(num_filters_base*2, (3,3), padding="same", name=f"{name}_d3_conv")(d3)
    d3 = BatchNormalization(name=f"{name}_d3_bn")(d3); d3 = Activation("relu", name=f"{name}_d3_relu")(d3)

    d4 = UpSampling2D((2,2), name=f"{name}_d4_upsample")(d3) # From 64x64 to 128x128
    d4 = concatenate([d4, e1], name=f"{name}_d4_concat") # Skip from e1 (128x128)
    d4 = Conv2D(num_filters_base, (3,3), padding="same", name=f"{name}_d4_conv")(d4)
    d4 = BatchNormalization(name=f"{name}_d4_bn")(d4); d4 = Activation("relu", name=f"{name}_d4_relu")(d4)

    output_image = Conv2D(1, (1,1), padding="same", activation="sigmoid", name=f"{name}_output_conv")(d4) # Sigmoid for 0-1 pixel values

    generator_model = Model(inputs=[outline_input, condition_input], outputs=output_image, name=name)
    return generator_model

def build_combined_model(img_shape=(64, 64, 1), num_examples=5, example_latent_dim=64, gen_filters_base=16):
    # Inputs
    example_img_inputs = [Input(shape=img_shape, name=f"example_img_input_{i}") for i in range(num_examples)]
    outline_input = Input(shape=img_shape, name="outline_main_input")

    # Create shared example encoder
    example_encoder = build_example_encoder(input_shape=img_shape, latent_dim=example_latent_dim)
    
    # Get features for each example image
    example_features = [example_encoder(img_input) for img_input in example_img_inputs]

    # Average the features
    if num_examples > 1:
        averaged_features = Average(name="average_example_features")(example_features)
    else: # Should not happen with num_examples=5, but good for flexibility
        averaged_features = example_features[0] 
        
    # Generator
    generator = build_generator(outline_input_shape=img_shape, example_latent_dim=example_latent_dim, num_filters_base=gen_filters_base)
    
    # Generate the image
    generated_output = generator([outline_input, averaged_features])

    combined_model = Model(inputs=example_img_inputs + [outline_input], outputs=generated_output, name="pixel_art_cat_generator")
    return combined_model, example_encoder, generator # Return sub-models for potential separate saving/loading

if __name__ == '__main__': # For testing the model structure
    IMG_SHAPE = (128,128,1)
    model, enc, gen = build_combined_model(IMG_SHAPE, num_examples=5)
    model.summary()
    # enc.summary()
    # gen.summary()
    # Check parameter counts:
    # Parameter counts for 128x128 with num_filters_base=16, latent_dim=64:
    # Example Encoder: ~37k
    # Generator: ~1.18M
    # Total: ~1.22M. Still well within <2M.