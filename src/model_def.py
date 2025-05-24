# src/model_def.py
import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    GlobalAveragePooling2D, Dense, Reshape, Add, Activation, BatchNormalization,
    Layer, Concatenate, LeakyReLU, LayerNormalization, MultiHeadAttention, Embedding,
    Conv2DTranspose  # Added for U-Net upsampling option
)
from tensorflow.keras.models import Model

IMG_DIM = 128
PATCH_SIZE = 16  # For StyleEncoder
NUM_PATCHES = (IMG_DIM // PATCH_SIZE) ** 2

# --- Custom FiLM Layer (Keep this for future use in Stage 3) ---


class FiLMLayer(Layer):
    def __init__(self, **kwargs):
        super(FiLMLayer, self).__init__(**kwargs)

    def call(self, inputs):
        if len(inputs) == 2:
            feature_map, combined_params = inputs
            num_channels_half = tf.shape(combined_params)[-1] // 2
            gamma = combined_params[..., :num_channels_half]
            beta = combined_params[..., num_channels_half:]
        elif len(inputs) == 3:
            feature_map, gamma, beta = inputs
        else:
            raise ValueError("FiLMLayer expects 2 or 3 inputs")
        gamma_r = tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1)
        beta_r = tf.expand_dims(tf.expand_dims(beta, axis=1), axis=1)
        return feature_map * gamma_r + beta_r

    def get_config(self):
        return super(FiLMLayer, self).get_config()

# --- Stage 1: Style Encoder (Transformer-based) ---


class PatchEmbed(Layer):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_chans=1, embed_dim=256, name="patch_embed", **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, padding="valid", name=f"{name}_proj")

    def call(self, x):  # x is [B, H, W, C]
        x = self.proj(x)  # [B, H/P, W/P, embed_dim]
        B, H_P, W_P, C = tf.shape(x)[0], tf.shape(
            x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H_P * W_P, C])  # [B, NumPatches, embed_dim]
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
        })
        return config


class StyleEncoder(Model):
    def __init__(self, num_examples=5, patch_size=16, img_dim_encoder=IMG_DIM,  # Pass img_dim for pos_embed
                 embed_dim=256, num_heads=4, num_transformer_layers=3,
                 num_style_tokens=5, name="style_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_examples = num_examples
        self.embed_dim = embed_dim  # Store embed_dim
        self.num_style_tokens = num_style_tokens

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

        # Use a different name to avoid conflict with Keras 'layers' property
        self.transformer_layers_list = []
        for i in range(num_transformer_layers):
            mha = MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1, name=f"mha_{i}")
            norm1 = LayerNormalization(epsilon=1e-6, name=f"norm1_{i}")
            # MLP block
            dense1 = Dense(embed_dim * 4, activation="gelu",
                           name=f"mlp_dense1_{i}")
            # No activation before add+norm
            dense2 = Dense(embed_dim, name=f"mlp_dense2_{i}")
            norm2 = LayerNormalization(epsilon=1e-6, name=f"norm2_{i}")
            self.transformer_layers_list.append(
                [mha, norm1, dense1, dense2, norm2])  # Store as a list of lists

        self.style_tokens_param = self.add_weight(
            name="style_tokens_learnable",  # Changed name to avoid potential conflict
            shape=(1, num_style_tokens, embed_dim),
            initializer="random_normal",
            trainable=True
        )

        # Calculate sequence length for positional embedding
        num_patches_per_example = (img_dim_encoder // patch_size) ** 2
        self.total_sequence_length = num_style_tokens + \
            (num_examples * num_patches_per_example)

        # Define Positional Embedding layer in __init__
        self.pos_embed_layer = Embedding(
            input_dim=self.total_sequence_length,
            output_dim=embed_dim,
            name="positional_embedding"
        )

    # Expects a list of 5 example tensors [B, H, W, 1]
    def call(self, examples_list):
        B = tf.shape(examples_list[0])[0]

        all_patches_list = []
        for i in range(self.num_examples):
            patches = self.patch_embed(examples_list[i])
            all_patches_list.append(patches)

        all_patches = Concatenate(
            axis=1, name="concat_example_patches")(all_patches_list)

        style_queries = tf.tile(self.style_tokens_param, [B, 1, 1])
        combined_input_sequence = Concatenate(axis=1, name="concat_queries_patches")([
            style_queries, all_patches])

        # Create positions and apply positional embedding
        # seq_len_dynamic = tf.shape(combined_input_sequence)[1] # This is dynamic
        # For Embedding layer, input_dim needs to be fixed at construction.
        # We use self.total_sequence_length which should match seq_len_dynamic
        positions = tf.range(
            start=0, limit=self.total_sequence_length, delta=1)
        positions_batched = tf.tile(tf.expand_dims(positions, 0), [
                                    B, 1])  # Tile positions for batch

        pos_embeddings = self.pos_embed_layer(
            positions_batched)  # Use pre-defined layer

        x = combined_input_sequence + pos_embeddings

        for mha, norm1, dense1, dense2, norm2 in self.transformer_layers_list:
            attn_output = mha(query=x, value=x, key=x)
            x = norm1(x + attn_output)
            # dense1 is mlp_ffn_dim, dense2 projects back
            mlp_output = dense2(dense1(x))
            x = norm2(x + mlp_output)

        return x[:, :self.num_style_tokens]

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_examples": self.num_examples,
            "patch_size": self.patch_embed.patch_size,  # Assuming PatchEmbed stores it
            # Reconstruct if needed
            "img_dim_encoder": self.pos_embed_layer.input_dim - self.num_style_tokens,
            "embed_dim": self.embed_dim,
            "num_heads": self.transformer_layers_list[0][0].num_heads if self.transformer_layers_list else 0,
            "num_transformer_layers": len(self.transformer_layers_list),
            "num_style_tokens": self.num_style_tokens,
        })
        return config

# --- Placeholder U-Net (will be replaced by Stage 2 & 3 later) ---
# This is a simplified U-Net that takes a style vector for basic conditioning.
# It's here so `train.py` can run and test the loss function.


def build_placeholder_unet_with_style_injection(
    outline_input_shape=(IMG_DIM, IMG_DIM, 1),
    style_vector_shape=None,
    num_filters_base=32,
    name="placeholder_unet_style"
):
    outline_input = Input(shape=outline_input_shape,
                          name=f"{name}_outline_input")
    style_input = Input(shape=style_vector_shape, name=f"{name}_style_input")

    # Encoder
    # Block 1
    e1 = Conv2D(num_filters_base, (4, 4), strides=2,
                padding="same", name=f"{name}_e1_conv")(outline_input)
    # No BatchNormalization on the first conv layer is a common practice
    e1 = LeakyReLU(alpha=0.2, name=f"{name}_e1_lrelu")(e1)  # 64x64

    # Block 2
    e2 = Conv2D(num_filters_base*2, (4, 4), strides=2,
                padding="same", name=f"{name}_e2_conv")(e1)
    e2 = BatchNormalization(name=f"{name}_e2_bn")(e2)
    e2 = LeakyReLU(alpha=0.2, name=f"{name}_e2_lrelu")(e2)  # 32x32

    # Block 3
    e3 = Conv2D(num_filters_base*4, (4, 4), strides=2,
                padding="same", name=f"{name}_e3_conv")(e2)
    e3 = BatchNormalization(name=f"{name}_e3_bn")(e3)
    e3 = LeakyReLU(alpha=0.2, name=f"{name}_e3_lrelu")(e3)  # 16x16

    # Block 4
    e4 = Conv2D(num_filters_base*8, (4, 4), strides=2,
                padding="same", name=f"{name}_e4_conv")(e3)
    e4 = BatchNormalization(name=f"{name}_e4_bn")(e4)
    e4 = LeakyReLU(alpha=0.2, name=f"{name}_e4_lrelu")(e4)  # 8x8

    # Bottleneck - Inject style here (very basic for placeholder)
    b = Conv2D(num_filters_base*8, (4, 4), strides=2,
               padding="same", name=f"{name}_b_conv")(e4)  # 4x4
    b = BatchNormalization(name=f"{name}_b_bn")(b)
    b = LeakyReLU(alpha=0.2, name=f"{name}_b_lrelu_bottleneck")(
        b)  # Use unique name

    style_processed = Dense(
        4 * 4 * b.shape[-1], name=f"{name}_style_dense_proj")(style_input)
    style_reshaped = Reshape(
        (4, 4, b.shape[-1]), name=f"{name}_style_reshape")(style_processed)
    b_conditioned = Add(name=f"{name}_style_add_bottleneck")(
        [b, style_reshaped])

    # Decoder
    # Block d1 (upsample to 8x8)
    d1 = Conv2DTranspose(num_filters_base*8, (4, 4), strides=2,
                         padding="same", name=f"{name}_d1_upconv")(b_conditioned)
    d1 = BatchNormalization(name=f"{name}_d1_bn")(d1)
    d1 = Activation("relu", name=f"{name}_d1_relu")(
        d1)  # Using ReLU in decoder is common
    d1 = concatenate([d1, e4], name=f"{name}_d1_concat")
    # Add a conv block after concat
    d1 = Conv2D(num_filters_base*8, (3, 3), padding="same",
                name=f"{name}_d1_conv_after_concat")(d1)
    d1 = BatchNormalization(name=f"{name}_d1_bn2")(d1)
    d1 = Activation("relu", name=f"{name}_d1_relu2")(d1)

    # Block d2 (upsample to 16x16)
    d_out = Conv2DTranspose(num_filters_base*4, (4, 4),
                            strides=2, padding="same", name=f"{name}_d2_upconv")(d1)
    d_out = BatchNormalization(name=f"{name}_d2_bn")(d_out)
    d_out = Activation("relu", name=f"{name}_d2_relu")(d_out)
    d_out = concatenate([d_out, e3], name=f"{name}_d2_concat")
    d_out = Conv2D(num_filters_base*4, (3, 3), padding="same",
                   name=f"{name}_d2_conv_after_concat")(d_out)
    d_out = BatchNormalization(name=f"{name}_d2_bn2")(d_out)
    d_out = Activation("relu", name=f"{name}_d2_relu2")(d_out)

    # Block d3 (upsample to 32x32)
    d_out = Conv2DTranspose(num_filters_base*2, (4, 4), strides=2,
                            padding="same", name=f"{name}_d3_upconv")(d_out)
    d_out = BatchNormalization(name=f"{name}_d3_bn")(d_out)
    d_out = Activation("relu", name=f"{name}_d3_relu")(d_out)
    d_out = concatenate([d_out, e2], name=f"{name}_d3_concat")
    d_out = Conv2D(num_filters_base*2, (3, 3), padding="same",
                   name=f"{name}_d3_conv_after_concat")(d_out)
    d_out = BatchNormalization(name=f"{name}_d3_bn2")(d_out)
    d_out = Activation("relu", name=f"{name}_d3_relu2")(d_out)

    # Block d4 (upsample to 64x64)
    d_out = Conv2DTranspose(num_filters_base, (4, 4), strides=2,
                            padding="same", name=f"{name}_d4_upconv")(d_out)
    d_out = BatchNormalization(name=f"{name}_d4_bn")(d_out)
    d_out = Activation("relu", name=f"{name}_d4_relu")(d_out)
    d_out = concatenate([d_out, e1], name=f"{name}_d4_concat")
    d_out = Conv2D(num_filters_base, (3, 3), padding="same",
                   name=f"{name}_d4_conv_after_concat")(d_out)
    d_out = BatchNormalization(name=f"{name}_d4_bn2")(d_out)
    d_out = Activation("relu", name=f"{name}_d4_relu2")(d_out)

    # Final output layer to get to 128x128
    final_up = Conv2DTranspose(num_filters_base // 2, (4, 4),
                               strides=2, padding="same", name=f"{name}_final_upconv")(d_out)
    # No BN before final activation if it's sigmoid/tanh
    final_up = Activation("relu", name=f"{name}_final_relu")(
        final_up)  # Or LeakyReLU
    output_image = Conv2D(1, (3, 3), activation='sigmoid',
                          padding='same', name=f"{name}_output_conv")(final_up)

    model = Model(inputs=[outline_input, style_input],
                  outputs=output_image, name=name)
    return model


# --- Combined Model (integrating Stage 1 StyleEncoder with Placeholder U-Net) ---
def build_combined_model_phase1(
    img_shape=(IMG_DIM, IMG_DIM, 1),
    num_examples=5,
    # StyleEncoder params
    patch_size_style=16,
    embed_dim_style=128,  # Smaller embed_dim for StyleEncoder initially
    num_heads_style=4,
    num_transformer_layers_style=2,  # Fewer layers for initial test
    num_style_tokens=5,
    # Placeholder U-Net params
    gen_filters_base=32
):
    example_img_inputs = [Input(
        shape=img_shape, name=f"example_img_input_{i}") for i in range(num_examples)]
    outline_input = Input(shape=img_shape, name="outline_main_input")

    # 1. Style Encoder
    style_encoder_model = StyleEncoder(
        num_examples=num_examples,
        patch_size=patch_size_style,
        embed_dim=embed_dim_style,
        num_heads=num_heads_style,
        num_transformer_layers=num_transformer_layers_style,
        num_style_tokens=num_style_tokens,
    )
    # Pass the list of Keras Tensors (Inputs) to the StyleEncoder Model
    # [B, num_style_tokens, embed_dim_style]
    style_tokens_output = style_encoder_model(example_img_inputs)

    # For placeholder U-Net, flatten the style tokens to a single vector
    # This is a simplification; Stage 2 (SpatialPlanner) will use these tokens more intelligently
    style_vector_flat = Reshape(
        (num_style_tokens * embed_dim_style,), name="flatten_style_tokens")(style_tokens_output)

    # 2. Placeholder U-Net (will be replaced by Stage 2 + Stage 3 later)
    placeholder_unet = build_placeholder_unet_with_style_injection(
        outline_input_shape=img_shape,
        style_vector_shape=(num_style_tokens * embed_dim_style,),
        num_filters_base=gen_filters_base
    )

    generated_output = placeholder_unet([outline_input, style_vector_flat])

    combined_model = Model(
        inputs=example_img_inputs + [outline_input],
        outputs=generated_output,
        name="cat_generator_phase1_style_encoder"
    )
    return combined_model


if __name__ == '__main__':
    # Test Stage 1 combined model
    model = build_combined_model_phase1(
        embed_dim_style=128,  # Keep small for testing
        num_transformer_layers_style=2,
        gen_filters_base=32
    )
    model.summary(line_length=150)
    print("Attempting to save and reload model for serializability check...")

    temp_model_path_keras = "temp_model_serial_check.keras"
    temp_model_path_h5 = "temp_model_serial_check.h5"

    # Define custom objects dictionary
    custom_objects_for_test = {
        'FiLMLayer': FiLMLayer,
        'PatchEmbed': PatchEmbed,
        'StyleEncoder': StyleEncoder
        # Add any other custom layers or models used
    }
    try:
        # Test Keras native format
        model.save(temp_model_path_keras)
        loaded_model_keras = tf.keras.models.load_model(
            temp_model_path_keras, custom_objects=custom_objects_for_test)
        print(
            f"Successfully saved and loaded Keras native format: {temp_model_path_keras}")
        os.remove(temp_model_path_keras)

        # Test H5 format
        model.save(temp_model_path_h5)
        loaded_model_h5 = tf.keras.models.load_model(
            temp_model_path_h5, custom_objects=custom_objects_for_test)
        print(f"Successfully saved and loaded H5 format: {temp_model_path_h5}")
        os.remove(temp_model_path_h5)

        print("SERIALIZABILITY CHECK PASSED!")
    except Exception as e:
        print(f"SERIALIZABILITY CHECK FAILED: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 50)

    # Test StyleEncoder standalone (optional)
    # print("\n--- StyleEncoder Standalone Test ---")
    # example_inputs_standalone = [Input(shape=(IMG_DIM,IMG_DIM,1)) for _ in range(5)]
    # style_enc = StyleEncoder(embed_dim=128, num_transformer_layers=2)
    # output_tokens = style_enc(example_inputs_standalone)
    # style_enc_model_standalone = Model(inputs=example_inputs_standalone, outputs=output_tokens)
    # style_enc_model_standalone.summary() # ~0.5M to 1M params depending on embed_dim and layers
