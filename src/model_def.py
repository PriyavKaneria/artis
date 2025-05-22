# src/model_def.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    GlobalAveragePooling2D, Dense, Reshape, Add, Activation, BatchNormalization,
    Layer, Concatenate, LeakyReLU, Lambda
)
from tensorflow.keras.models import Model

IMG_DIM = 128 # Should be consistent with prepare_data.py and train.py

# --- Custom FiLM Layer ---
class FiLMLayer(Layer):
    """Applies Feature-wise Linear Modulation (FiLM)."""
    def __init__(self, **kwargs):
        super(FiLMLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Now expects a list: [feature_map, combined_gamma_beta_params]
        # OR: [feature_map, gamma, beta] if split before calling
        # Let's design it to take combined params and split internally
        if len(inputs) == 2: # feature_map and combined_params
            feature_map, combined_params = inputs
            # Split gamma and beta from combined_params
            # The number of channels for gamma/beta is half the size of combined_params' last dimension
            num_channels = tf.shape(combined_params)[-1] // 2 # Get dynamically
            gamma = combined_params[..., :num_channels]
            beta = combined_params[..., num_channels:]
        elif len(inputs) == 3: # feature_map, gamma, beta (already split)
            feature_map, gamma, beta = inputs
        else:
            raise ValueError("FiLMLayer expects 2 or 3 inputs: [feature_map, combined_params] or [feature_map, gamma, beta]")

        # Reshape gamma and beta to be broadcastable: (B, 1, 1, C)
        gamma_reshaped = tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1)
        beta_reshaped = tf.expand_dims(tf.expand_dims(beta, axis=1), axis=1)
        
        return feature_map * gamma_reshaped + beta_reshaped

    def get_config(self):
        config = super(FiLMLayer, self).get_config()
        return config

# --- Example Encoder ---
def build_example_encoder(input_shape=(IMG_DIM, IMG_DIM, 1), latent_dim=256, name="example_encoder"):
    img_input = Input(shape=input_shape, name=f"{name}_input")
    
    x = Conv2D(32, (5, 5), strides=(2, 2), padding="same", name=f"{name}_conv1")(img_input) # 64x64
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = LeakyReLU(alpha=0.2, name=f"{name}_lrelu1")(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv2")(x) # 32x32
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = LeakyReLU(alpha=0.2, name=f"{name}_lrelu2")(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv3")(x) # 16x16
    x = BatchNormalization(name=f"{name}_bn3")(x)
    x = LeakyReLU(alpha=0.2, name=f"{name}_lrelu3")(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv4")(x) # 8x8
    x = BatchNormalization(name=f"{name}_bn4")(x)
    x = LeakyReLU(alpha=0.2, name=f"{name}_lrelu4")(x)
    
    # Global Average Pooling to get a feature vector
    feature_vector = GlobalAveragePooling2D(name=f"{name}_gap")(x) # Output shape (None, 256)
    
    # Optional: Dense layer to further process/project the feature vector
    processed_vector = Dense(latent_dim, activation='linear', name=f"{name}_dense_latent")(feature_vector)
    
    encoder = Model(img_input, processed_vector, name=name)
    return encoder

# --- Example Feature Aggregator (Concatenation + MLP) ---
def build_example_feature_aggregator(num_examples=5, example_latent_dim=256, style_vector_dim=512, name="feature_aggregator"):
    input_features = [Input(shape=(example_latent_dim,), name=f"{name}_example_feat_{i}") for i in range(num_examples)]
    
    if num_examples > 1:
        concatenated_features = Concatenate(name=f"{name}_concat")(input_features) # Shape (None, num_examples * example_latent_dim)
    else:
        concatenated_features = input_features[0]

    x = Dense(512, name=f"{name}_dense1")(concatenated_features)
    x = LeakyReLU(alpha=0.2, name=f"{name}_lrelu1")(x)
    x = Dense(style_vector_dim, activation='linear', name=f"{name}_style_vector_output")(x) # Final style vector
    
    aggregator = Model(inputs=input_features, outputs=x, name=name)
    return aggregator

# --- FiLM Parameter Predictor (a small MLP) ---
def build_film_param_predictor(style_vector_dim, num_target_channels, name_prefix="film_predictor"):
    # This function returns a Keras Model (MLP) that predicts gamma and beta
    # for a specific FiLM layer.
    style_input = Input(shape=(style_vector_dim,), name=f"{name_prefix}_style_input")
    
    x = Dense(256, name=f"{name_prefix}_dense1")(style_input) # Small hidden layer
    x = LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu1")(x)
    
    # Output 2 * num_target_channels (for gamma and beta)
    # Linear activation for gamma and beta is common. Gamma often initialized around 1, beta around 0.
    # Here, default initialization should be okay.
    film_params = Dense(num_target_channels * 2, activation='linear', name=f"{name_prefix}_params_output")(x)
    
    predictor_model = Model(inputs=style_input, outputs=film_params, name=name_prefix)
    return predictor_model


# --- U-Net Generator with FiLM ---

def build_generator_unet_with_film(
    outline_input_shape=(IMG_DIM, IMG_DIM, 1), 
    style_vector_dim=512, 
    num_filters_base=32,
    name="generator_unet_film"):

    outline_input = Input(shape=outline_input_shape, name=f"{name}_outline_input")
    style_vector_input = Input(shape=(style_vector_dim,), name=f"{name}_style_vector_input")

    film_predictors = {}

    def apply_film(feature_map, style_vec, layer_name_suffix):
        num_channels = feature_map.shape[-1] # Static shape from feature_map
        predictor_name = f"film_predictor_{layer_name_suffix}"
        
        if predictor_name not in film_predictors:
            film_predictors[predictor_name] = build_film_param_predictor(
                style_vector_dim, num_channels, name_prefix=predictor_name
            )
        
        params = film_predictors[predictor_name](style_vec) # params is (B, 2*C)
        
        # Pass the combined params to FiLMLayer, it will split
        return FiLMLayer(name=f"film_apply_{layer_name_suffix}")([feature_map, params])

    # -- Encoder Path (on outline_input) --
    # Block 1: 128 -> 64
    e1 = Conv2D(num_filters_base, (4,4), strides=2, padding="same", name=f"{name}_e1_conv")(outline_input)
    e1 = LeakyReLU(alpha=0.2, name=f"{name}_e1_lrelu")(e1) # No BN on first encoder layer often
    
    # Block 2: 64 -> 32
    e2 = Conv2D(num_filters_base*2, (4,4), strides=2, padding="same", name=f"{name}_e2_conv")(e1)
    e2 = BatchNormalization(name=f"{name}_e2_bn")(e2); e2 = LeakyReLU(alpha=0.2, name=f"{name}_e2_lrelu")(e2)
    
    # Block 3: 32 -> 16
    e3 = Conv2D(num_filters_base*4, (4,4), strides=2, padding="same", name=f"{name}_e3_conv")(e2)
    e3 = BatchNormalization(name=f"{name}_e3_bn")(e3); e3 = LeakyReLU(alpha=0.2, name=f"{name}_e3_lrelu")(e3)

    # Block 4: 16 -> 8
    e4 = Conv2D(num_filters_base*8, (4,4), strides=2, padding="same", name=f"{name}_e4_conv")(e3)
    e4 = BatchNormalization(name=f"{name}_e4_bn")(e4); e4 = LeakyReLU(alpha=0.2, name=f"{name}_e4_lrelu")(e4)
    
    # Block 5: 8 -> 4 (Bottleneck Entry)
    e5 = Conv2D(num_filters_base*8, (4,4), strides=2, padding="same", name=f"{name}_e5_conv")(e4) # Smallest spatial dim: 4x4
    e5 = BatchNormalization(name=f"{name}_e5_bn")(e5); e5 = LeakyReLU(alpha=0.2, name=f"{name}_e5_lrelu")(e5)

    # -- Bottleneck --
    # No skip connection directly from bottleneck in typical pix2pix-like U-Nets
    b = Conv2D(num_filters_base*8, (4,4), strides=2, padding="same", name=f"{name}_b_conv")(e5) # 2x2
    b = BatchNormalization(name=f"{name}_b_bn")(b); b = LeakyReLU(alpha=0.2, name=f"{name}_b_relu")(b)
    # Apply FiLM to bottleneck output before upsampling
    b_filmed = apply_film(b, style_vector_input, "bottleneck")

    # -- Decoder Path (with skip connections and FiLM) --
    # Upsample from 2x2 to 4x4
    d0 = UpSampling2D(size=(2,2), interpolation='nearest', name=f"{name}_d0_upsample")(b_filmed)
    d0 = Conv2D(num_filters_base*8, (3,3), padding="same", name=f"{name}_d0_conv")(d0)
    d0 = BatchNormalization(name=f"{name}_d0_bn")(d0); d0 = Activation("relu", name=f"{name}_d0_relu")(d0)
    d0 = concatenate([d0, e5], name=f"{name}_d0_concat") # Skip from e5 (4x4)
    d0_filmed = apply_film(d0, style_vector_input, "d0")
    
    # Upsample from 4x4 to 8x8
    d1 = UpSampling2D(size=(2,2), interpolation='nearest', name=f"{name}_d1_upsample")(d0_filmed)
    d1 = Conv2D(num_filters_base*8, (3,3), padding="same", name=f"{name}_d1_conv")(d1)
    d1 = BatchNormalization(name=f"{name}_d1_bn")(d1); d1 = Activation("relu", name=f"{name}_d1_relu")(d1)
    d1 = concatenate([d1, e4], name=f"{name}_d1_concat") # Skip from e4 (8x8)
    d1_filmed = apply_film(d1, style_vector_input, "d1")

    # Upsample from 8x8 to 16x16
    d2 = UpSampling2D(size=(2,2), interpolation='nearest', name=f"{name}_d2_upsample")(d1_filmed)
    d2 = Conv2D(num_filters_base*4, (3,3), padding="same", name=f"{name}_d2_conv")(d2)
    d2 = BatchNormalization(name=f"{name}_d2_bn")(d2); d2 = Activation("relu", name=f"{name}_d2_relu")(d2)
    d2 = concatenate([d2, e3], name=f"{name}_d2_concat")
    d2_filmed = apply_film(d2, style_vector_input, "d2")
    
    # Upsample from 16x16 to 32x32
    d3 = UpSampling2D(size=(2,2), interpolation='nearest', name=f"{name}_d3_upsample")(d2_filmed)
    d3 = Conv2D(num_filters_base*2, (3,3), padding="same", name=f"{name}_d3_conv")(d3)
    d3 = BatchNormalization(name=f"{name}_d3_bn")(d3); d3 = Activation("relu", name=f"{name}_d3_relu")(d3)
    d3 = concatenate([d3, e2], name=f"{name}_d3_concat")
    d3_filmed = apply_film(d3, style_vector_input, "d3")

    # Upsample from 32x32 to 64x64
    d4 = UpSampling2D(size=(2,2), interpolation='nearest', name=f"{name}_d4_upsample")(d3_filmed)
    d4 = Conv2D(num_filters_base, (3,3), padding="same", name=f"{name}_d4_conv")(d4)
    d4 = BatchNormalization(name=f"{name}_d4_bn")(d4); d4 = Activation("relu", name=f"{name}_d4_relu")(d4)
    d4 = concatenate([d4, e1], name=f"{name}_d4_concat")
    d4_filmed = apply_film(d4, style_vector_input, "d4")

    # Upsample from 64x64 to 128x128 (Output Layer)
    output_image_pre_act = UpSampling2D(size=(2,2), interpolation='nearest', name=f"{name}_output_upsample")(d4_filmed)
    output_image_pre_act = Conv2D(num_filters_base // 2, (3,3), padding="same", name=f"{name}_output_conv1")(output_image_pre_act) # Reduce channels
    output_image_pre_act = LeakyReLU(alpha=0.2, name=f"{name}_output_lrelu")(output_image_pre_act)
    
    # Final convolution to get 1 channel for grayscale, with tanh activation (output -1 to 1)
    # Scaled later to 0-1 if needed, or use sigmoid if strict 0-1 is desired from model
    output_image = Conv2D(1, (3,3), padding="same", activation="sigmoid", name=f"{name}_output_final_conv")(output_image_pre_act) # Sigmoid for 0-1 output

    generator_model = Model(inputs=[outline_input, style_vector_input], outputs=output_image, name=name)
    
    # The film_predictors dict is not directly part of the Keras model's layers,
    # but their weights will be trained as they are called within the generator_model's graph.
    # We return it just for inspection if needed, but it's not used to build the final combined model.
    return generator_model, film_predictors


# --- Combined Model ---
def build_combined_film_model(
    img_shape=(IMG_DIM, IMG_DIM, 1), 
    num_examples=5, 
    example_latent_dim=256, 
    style_vector_dim=512,
    gen_filters_base=32):

    example_img_inputs = [Input(shape=img_shape, name=f"example_img_input_{i}") for i in range(num_examples)]
    outline_input = Input(shape=img_shape, name="outline_main_input")
    example_encoder_model = build_example_encoder(input_shape=img_shape, latent_dim=example_latent_dim)
    individual_example_features = [example_encoder_model(img_input) for img_input in example_img_inputs]
    feature_aggregator_model = build_example_feature_aggregator(
        num_examples=num_examples, 
        example_latent_dim=example_latent_dim, 
        style_vector_dim=style_vector_dim
    )
    c_style_vector = feature_aggregator_model(individual_example_features)
    generator_model, _ = build_generator_unet_with_film(
        outline_input_shape=img_shape,
        style_vector_dim=style_vector_dim,
        num_filters_base=gen_filters_base
    )
    generated_output = generator_model([outline_input, c_style_vector])
    combined_model = Model(
        inputs=example_img_inputs + [outline_input], 
        outputs=generated_output, 
        name="cat_generator_with_film"
    )
    return combined_model #, example_encoder_model, feature_aggregator_model, generator_model # Optionally return sub-models

if __name__ == '__main__':
    # For testing the model structure and parameter count
    IMG_SHAPE_TEST = (IMG_DIM, IMG_DIM, 1)
    NUM_EXAMPLES_TEST = 5
    EXAMPLE_LATENT_DIM_TEST = 256
    STYLE_VECTOR_DIM_TEST = 512
    GEN_FILTERS_BASE_TEST = 32 # Try 32 for a start, can go to 48 or 64 if budget allows

    model = build_combined_film_model(
        img_shape=IMG_SHAPE_TEST,
        num_examples=NUM_EXAMPLES_TEST,
        example_latent_dim=EXAMPLE_LATENT_DIM_TEST,
        style_vector_dim=STYLE_VECTOR_DIM_TEST,
        gen_filters_base=GEN_FILTERS_BASE_TEST
    )
    model.summary(line_length=150) # Print summary with wider lines

    # Check parameter counts of sub-components if needed by building them separately:
    # enc = build_example_encoder(input_shape=IMG_SHAPE_TEST, latent_dim=EXAMPLE_LATENT_DIM_TEST)
    # print("\nExample Encoder Summary:")
    # enc.summary() # Around 0.5M with latent_dim=256

    # agg_inputs = [tf.keras.Input(shape=(EXAMPLE_LATENT_DIM_TEST,)) for _ in range(NUM_EXAMPLES_TEST)]
    # agg = build_example_feature_aggregator(num_examples=NUM_EXAMPLES_TEST, example_latent_dim=EXAMPLE_LATENT_DIM_TEST, style_vector_dim=STYLE_VECTOR_DIM_TEST)
    # # To see summary for aggregator, you'd need to call it: _ = agg(agg_inputs)
    # print("\nFeature Aggregator Summary (approx based on Dense layers):")
    # # Example: (5*256)*512 + 512 + 512*512 + 512  ~ 0.65M + 0.26M ~ 0.9M
    # print(f"Aggregator approx params: {(NUM_EXAMPLES_TEST*EXAMPLE_LATENT_DIM_TEST)*512 + 512 + 512*STYLE_VECTOR_DIM_TEST + STYLE_VECTOR_DIM_TEST}")


    # gen, film_preds_dict = build_generator_unet_with_film(
    #     outline_input_shape=IMG_SHAPE_TEST, 
    #     style_vector_dim=STYLE_VECTOR_DIM_TEST, 
    #     num_filters_base=GEN_FILTERS_BASE_TEST
    # )
    # print("\nGenerator U-Net with FiLM Summary:")
    # gen.summary(line_length=150)
    # print(f"\nNumber of FiLM predictor MLPs created: {len(film_preds_dict)}")
    # for name, predictor in film_preds_dict.items():
    #     print(f"  {name} params: {predictor.count_params()}")

    # With GEN_FILTERS_BASE_TEST = 32:
    # Example Encoder: ~0.5 M
    # Aggregator MLP: ~(5*256)*512 + 512 (bias) + 512*512 + 512 (bias) = 655360 + 512 + 262144 + 512 = ~0.92 M
    # U-Net Generator (num_filters_base=32): ~6.8 M (this is the main part)
    # FiLM Predictor MLPs (total for 6 FiLM layers): ~0.6 M
    # Total params: ~0.5 + ~0.92 + ~6.8 + ~0.6 = ~8.82M. This is within the 10M target.

    # If GEN_FILTERS_BASE_TEST = 24:
    # U-Net Generator (num_filters_base=24): ~3.8 M
    # FiLM Predictor MLPs (total for 6 FiLM layers, channels reduced): ~0.4 M
    # Total params with base=24: ~0.5 + ~0.92 + ~3.8 + ~0.4 = ~5.62M. (Safer start)