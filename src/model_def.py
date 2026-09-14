# src/model_def.py
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    GlobalAveragePooling2D, Dense, Reshape, Add, Activation, BatchNormalization,
    Layer, Concatenate, LeakyReLU, LayerNormalization, MultiHeadAttention, Embedding,
    Conv2DTranspose, AveragePooling2D, GlobalAveragePooling1D, ReLU
)
from tensorflow.keras.models import Model
import os # For serializability check
from tensorflow.keras.losses import Loss

IMG_DIM = 128 
PATCH_SIZE_STYLE = 16

class FiLMLayer(Layer):
    def __init__(self, name="film_layer", **kwargs):
        super(FiLMLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        # Strictly expect a list/tuple of 2 inputs: [feature_map, combined_gamma_beta_params]
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError(
                f"FiLMLayer expects a list/tuple of 2 inputs: [feature_map, combined_gamma_beta_params]. "
                f"Received: {inputs} of type {type(inputs)} with length {len(inputs) if isinstance(inputs, (list, tuple)) else 'N/A'}"
            )
        
        feature_map, combined_params = inputs # Direct unpacking
        
        num_total_params = tf.shape(combined_params)[-1]
        num_channels_half = num_total_params // 2

        gamma = combined_params[..., :num_channels_half]
        beta = combined_params[..., num_channels_half:]
        
        gamma_reshaped = tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1)
        beta_reshaped = tf.expand_dims(tf.expand_dims(beta, axis=1), axis=1)
        
        return feature_map * gamma_reshaped + beta_reshaped

    def get_config(self):
        config = super(FiLMLayer, self).get_config()
        return config

# --- Stage 1: Style Encoder (Transformer-based) ---
class PatchEmbed(Layer):
    def __init__(self, patch_size=16, in_chans=1, embed_dim=256, name="patch_embed", **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid", name=f"{self.name}_proj")
    def call(self, x):
        x = self.proj(x)
        B, H_P, W_P, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B, H_P * W_P, C])
        return x
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "in_chans": self.in_chans, "embed_dim": self.embed_dim})
        return config

class StyleEncoder(Model):
    def __init__(self, num_examples=5, patch_size=PATCH_SIZE_STYLE, img_dim_encoder=IMG_DIM,
                 embed_dim=256, num_heads=4, num_transformer_layers=3, 
                 num_style_tokens=5, name="style_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_examples = num_examples
        self.patch_size = patch_size
        self.img_dim_encoder = img_dim_encoder
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.num_style_tokens = num_style_tokens

        self.patch_embed = PatchEmbed(patch_size=self.patch_size, in_chans=1, embed_dim=self.embed_dim, name=f"{self.name}_patch_embed")
        
        self.transformer_blocks = []
        for i in range(self.num_transformer_layers):
            mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim // self.num_heads, dropout=0.1, name=f"{self.name}_mha_{i}")
            norm1 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm1_{i}")
            dense_ffn1 = Dense(self.embed_dim * 4, name=f"{self.name}_ffn1_{i}") # No activation here, separate
            dense_ffn2 = Dense(self.embed_dim, name=f"{self.name}_ffn2_{i}")
            norm2 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm2_{i}")
            self.transformer_blocks.append({
                "mha": mha, "norm1": norm1, "ffn1": dense_ffn1,
                "ffn2": dense_ffn2, "norm2": norm2, "gelu": Activation('gelu', name=f"{self.name}_gelu_{i}")
            })
            
        self.style_tokens_param = self.add_weight(
            name="style_tokens_learnable", shape=(1, self.num_style_tokens, self.embed_dim),
            initializer="random_normal", trainable=True
        )
        
        num_patches_per_example = (self.img_dim_encoder // self.patch_size) ** 2
        self.total_sequence_length = self.num_style_tokens + (self.num_examples * num_patches_per_example)
        self.pos_embed_layer = Embedding(
            input_dim=self.total_sequence_length, output_dim=self.embed_dim, name=f"{self.name}_pos_embed"
        )

    def call(self, examples_list): # List of example tensors [B, H, W, 1]
        B = tf.shape(examples_list[0])[0]
        
        all_patches_list = [self.patch_embed(examples_list[i]) for i in range(self.num_examples)]
        all_patches = Concatenate(axis=1, name=f"{self.name}_concat_patches")(all_patches_list)
        
        style_queries = tf.tile(self.style_tokens_param, [B, 1, 1])
        combined_input_sequence = Concatenate(axis=1, name=f"{self.name}_concat_queries_patches")([style_queries, all_patches])
        
        positions = tf.range(start=0, limit=self.total_sequence_length, delta=1)
        positions_batched = tf.tile(tf.expand_dims(positions, 0), [B, 1])
        pos_embeddings = self.pos_embed_layer(positions_batched)
        x = combined_input_sequence + pos_embeddings
        
        for block_layers in self.transformer_blocks:
            attn_output = block_layers["mha"](query=x, value=x, key=x)
            x = block_layers["norm1"](x + attn_output)
            ffn_hidden = block_layers["ffn1"](x)
            ffn_hidden = block_layers["gelu"](ffn_hidden) # Separate GELU
            ffn_output = block_layers["ffn2"](ffn_hidden)
            x = block_layers["norm2"](x + ffn_output)
            
        return x[:, :self.num_style_tokens]

    def get_config(self):
        config = super().get_config() #Necessary for Model subclass
        config.update({
            "num_examples": self.num_examples, "patch_size": self.patch_size,
            "img_dim_encoder": self.img_dim_encoder, "embed_dim": self.embed_dim,
            "num_heads": self.num_heads, "num_transformer_layers": self.num_transformer_layers,
            "num_style_tokens": self.num_style_tokens
        })
        return config
    @classmethod
    def from_config(cls, config): # Necessary for Model subclass with custom args
        # PatchEmbed is created internally, no need to deserialize it separately here if it's not passed to __init__
        return cls(**config)


# --- Stage 2: Spatial Planner (CNN + Cross-Attention) ---
class SpatialPlanner(Model):
    def __init__(self, style_token_dim=256, plan_channels=128, num_attn_heads=4, name="spatial_planner", **kwargs):
        super().__init__(name=name, **kwargs)
        self.style_token_dim = style_token_dim
        self.plan_channels = plan_channels # C_plan
        self.num_attn_heads = num_attn_heads

        # Outline Encoder CNN
        self.outline_conv1 = Conv2D(32, (7,7), padding="same", name=f"{self.name}_o_conv1")
        self.outline_gn1 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_o_gn1") # GroupNorm not std Keras, using LayerNorm for simplicity
        self.outline_relu1 = ReLU(name=f"{self.name}_o_relu1")
        self.outline_conv2 = Conv2D(64, (5,5), padding="same", name=f"{self.name}_o_conv2")
        self.outline_gn2 = LayerNormalization(epsilon=1e-6, name=f"{self.name}_o_gn2")
        self.outline_relu2 = ReLU(name=f"{self.name}_o_relu2")
        self.outline_conv3 = Conv2D(self.plan_channels, (3,3), padding="same", name=f"{self.name}_o_conv3_plan") # Output C_plan channels

        # Style Projection and Cross-Attention
        self.style_proj = Dense(self.plan_channels, name=f"{self.name}_style_proj") # Project style_tokens to plan_channels
        self.cross_attention = MultiHeadAttention(num_heads=self.num_attn_heads, key_dim=self.plan_channels // self.num_attn_heads, name=f"{self.name}_cross_attn")
        self.output_norm = LayerNormalization(epsilon=1e-6, name=f"{self.name}_output_norm")

    def call(self, inputs):
        style_tokens, outline_mask = inputs # style_tokens: [B, N_style_tokens, D_style], outline_mask: [B, H, W, 1]

        # 1. Process Outline
        s = self.outline_conv1(outline_mask)
        s = self.outline_gn1(s); s = self.outline_relu1(s)
        s = self.outline_conv2(s)
        s = self.outline_gn2(s); s = self.outline_relu2(s)
        spatial_features = self.outline_conv3(s) # [B, H, W, C_plan]
        
        B, H, W, C_plan = tf.shape(spatial_features)[0], tf.shape(spatial_features)[1], tf.shape(spatial_features)[2], tf.shape(spatial_features)[3]

        # 2. Prepare for Cross-Attention
        # Query: spatial_features (flattened)
        # Key/Value: style_tokens (projected)
        spatial_flat = tf.reshape(spatial_features, [B, H * W, C_plan]) # [B, HW, C_plan]
        projected_style_tokens = self.style_proj(style_tokens) # [B, N_style_tokens, C_plan]

        # 3. Cross-Attention: Each spatial location attends to style tokens
        # MHA expects query, value, key
        # planned_feats_flat = self.cross_attention(query=spatial_flat, value=projected_style_tokens, key=projected_style_tokens)
        # Add & Norm structure for attention output is common
        attn_output = self.cross_attention(query=spatial_flat, value=projected_style_tokens, key=projected_style_tokens)
        planned_feats_flat = self.output_norm(spatial_flat + attn_output) # Add residual connection from spatial_flat

        # 4. Reshape back to spatial
        planned_features_spatial = tf.reshape(planned_feats_flat, [B, H, W, C_plan])
        return planned_features_spatial # [B, H, W, C_plan]

    def get_config(self):
        config = super().get_config()
        config.update({
            "style_token_dim": self.style_token_dim, "plan_channels": self.plan_channels,
            "num_attn_heads": self.num_attn_heads
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# --- Stage 3: Adaptive Generator (U-Net like with FiLM Bottleneck & Plan Integration) ---
class AdaptiveBottleneck(Layer):
    def __init__(self, bottleneck_channels, style_token_dim, num_style_tokens, name="adaptive_bottleneck", **kwargs):
        super().__init__(name=name, **kwargs)
        self.bottleneck_channels = bottleneck_channels
        self.style_token_dim = style_token_dim # Dim of one style token from StyleEncoder
        self.num_style_tokens = num_style_tokens

        # Pool style tokens (e.g., mean pooling)
        self.style_pool = GlobalAveragePooling1D(keepdims=False, name=f"{self.name}_style_pool") # From [B, N_tokens, D_style] to [B, D_style]
        
        # MLP to predict FiLM params (gamma, beta) for bottleneck features
        self.film_param_predictor = Dense(bottleneck_channels * 2, name=f"{self.name}_film_mlp")
        self.film_layer = FiLMLayer(name=f"{self.name}_film_apply")

    def call(self, inputs):
        bottleneck_features, style_tokens = inputs # bottleneck_features: [B,H_b,W_b,C_b], style_tokens: [B, N_style_tokens, D_style]
        
        pooled_style_vector = self.style_pool(style_tokens) # [B, D_style]
        
        film_params = self.film_param_predictor(pooled_style_vector) # [B, C_b*2]
        # Gamma and Beta are split inside FiLMLayer now based on its new design
        
        modulated_bottleneck = self.film_layer([bottleneck_features, film_params])
        return modulated_bottleneck

    def get_config(self):
        config = super().get_config()
        config.update({
            "bottleneck_channels": self.bottleneck_channels,
            "style_token_dim": self.style_token_dim,
            "num_style_tokens": self.num_style_tokens
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AdaptiveGenerator(Model):
    def __init__(self, initial_plan_channels=128, num_filters_base=32, 
                 style_token_dim=256, num_style_tokens_for_bottleneck=5, 
                 img_dim_generator=IMG_DIM, name="adaptive_generator", **kwargs):
        super().__init__(name=name, **kwargs)
        self.initial_plan_channels = initial_plan_channels
        self.nfb = num_filters_base
        self.img_dim_generator = img_dim_generator

        # --- Encoder Layers ---
        # Enc1: Input (e.g., 128x128), Output (64x64)
        self.enc1_conv1 = Conv2D(self.nfb, (3,3), strides=1, padding="same", name=f"{self.name}_e1_c1")
        self.enc1_lrelu1 = LeakyReLU(alpha=0.2, name=f"{self.name}_e1_lrelu1")
        self.enc1_conv2_down = Conv2D(self.nfb, (3,3), strides=2, padding="same", name=f"{self.name}_e1_c2_down")
        self.enc1_bn2 = BatchNormalization(name=f"{self.name}_e1_bn2")
        self.enc1_lrelu2 = LeakyReLU(alpha=0.2, name=f"{self.name}_e1_lrelu2")

        # Enc2: Input (64x64), Output (32x32)
        self.enc2_conv1 = Conv2D(self.nfb*2, (3,3), strides=1, padding="same", name=f"{self.name}_e2_c1")
        self.enc2_bn1 = BatchNormalization(name=f"{self.name}_e2_bn1")
        self.enc2_lrelu1 = LeakyReLU(alpha=0.2, name=f"{self.name}_e2_lrelu1")
        self.enc2_conv2_down = Conv2D(self.nfb*2, (3,3), strides=2, padding="same", name=f"{self.name}_e2_c2_down")
        self.enc2_bn2 = BatchNormalization(name=f"{self.name}_e2_bn2")
        self.enc2_lrelu2 = LeakyReLU(alpha=0.2, name=f"{self.name}_e2_lrelu2")

        # Enc3: Input (32x32), Output (16x16)
        self.enc3_conv1 = Conv2D(self.nfb*4, (3,3), strides=1, padding="same", name=f"{self.name}_e3_c1")
        self.enc3_bn1 = BatchNormalization(name=f"{self.name}_e3_bn1")
        self.enc3_lrelu1 = LeakyReLU(alpha=0.2, name=f"{self.name}_e3_lrelu1")
        self.enc3_conv2_down = Conv2D(self.nfb*4, (3,3), strides=2, padding="same", name=f"{self.name}_e3_c2_down")
        self.enc3_bn2 = BatchNormalization(name=f"{self.name}_e3_bn2")
        self.enc3_lrelu2 = LeakyReLU(alpha=0.2, name=f"{self.name}_e3_lrelu2")

        # Enc4: Input (16x16), Output (8x8) - This is input to AdaptiveBottleneck
        self.enc4_conv1 = Conv2D(self.nfb*8, (3,3), strides=1, padding="same", name=f"{self.name}_e4_c1")
        self.enc4_bn1 = BatchNormalization(name=f"{self.name}_e4_bn1")
        self.enc4_lrelu1 = LeakyReLU(alpha=0.2, name=f"{self.name}_e4_lrelu1")
        self.enc4_conv2_down = Conv2D(self.nfb*8, (3,3), strides=2, padding="same", name=f"{self.name}_e4_c2_down")
        self.enc4_bn2 = BatchNormalization(name=f"{self.name}_e4_bn2")
        self.enc4_lrelu2 = LeakyReLU(alpha=0.2, name=f"{self.name}_e4_lrelu2")
        
        self.adaptive_bottleneck = AdaptiveBottleneck(
            bottleneck_channels=self.nfb*8, 
            style_token_dim=style_token_dim,
            num_style_tokens=num_style_tokens_for_bottleneck,
            name=f"{self.name}_adaptive_bneck"
        )

        # --- Decoder Layers ---
        self.pool = AveragePooling2D((2,2), name=f"{self.name}_avg_pool_skip")

        # Dec3 (Input from bottleneck 8x8, outputs 16x16)
        self.dec3_upconv = Conv2DTranspose(self.nfb*4, (4,4), strides=2, padding="same", name=f"{self.name}_d3_upconv")
        self.dec3_bn1 = BatchNormalization(name=f"{self.name}_d3_bn1"); self.dec3_relu1 = Activation("relu", name=f"{self.name}_d3_relu1")
        self.dec3_conv1 = Conv2D(self.nfb*4, (3,3), padding="same", name=f"{self.name}_d3_conv1")
        self.dec3_bn2 = BatchNormalization(name=f"{self.name}_d3_bn2"); self.dec3_relu2 = Activation("relu", name=f"{self.name}_d3_relu2")
        self.dec3_conv2 = Conv2D(self.nfb*4, (3,3), padding="same", name=f"{self.name}_d3_conv2") # Added for more capacity
        self.dec3_bn3 = BatchNormalization(name=f"{self.name}_d3_bn3"); self.dec3_relu3 = Activation("relu", name=f"{self.name}_d3_relu3")


        # Dec2 (Input from d3 16x16, outputs 32x32)
        self.dec2_upconv = Conv2DTranspose(self.nfb*2, (4,4), strides=2, padding="same", name=f"{self.name}_d2_upconv")
        self.dec2_bn1 = BatchNormalization(name=f"{self.name}_d2_bn1"); self.dec2_relu1 = Activation("relu", name=f"{self.name}_d2_relu1")
        self.dec2_conv1 = Conv2D(self.nfb*2, (3,3), padding="same", name=f"{self.name}_d2_conv1")
        self.dec2_bn2 = BatchNormalization(name=f"{self.name}_d2_bn2"); self.dec2_relu2 = Activation("relu", name=f"{self.name}_d2_relu2")
        self.dec2_conv2 = Conv2D(self.nfb*2, (3,3), padding="same", name=f"{self.name}_d2_conv2")
        self.dec2_bn3 = BatchNormalization(name=f"{self.name}_d2_bn3"); self.dec2_relu3 = Activation("relu", name=f"{self.name}_d2_relu3")

        # Dec1 (Input from d2 32x32, outputs 64x64)
        self.dec1_upconv = Conv2DTranspose(self.nfb, (4,4), strides=2, padding="same", name=f"{self.name}_d1_upconv")
        self.dec1_bn1 = BatchNormalization(name=f"{self.name}_d1_bn1"); self.dec1_relu1 = Activation("relu", name=f"{self.name}_d1_relu1")
        self.dec1_conv1 = Conv2D(self.nfb, (3,3), padding="same", name=f"{self.name}_d1_conv1")
        self.dec1_bn2 = BatchNormalization(name=f"{self.name}_d1_bn2"); self.dec1_relu2 = Activation("relu", name=f"{self.name}_d1_relu2")
        self.dec1_conv2 = Conv2D(self.nfb, (3,3), padding="same", name=f"{self.name}_d1_conv2")
        self.dec1_bn3 = BatchNormalization(name=f"{self.name}_d1_bn3"); self.dec1_relu3 = Activation("relu", name=f"{self.name}_d1_relu3")
        
        self.final_up = Conv2DTranspose(max(16, self.nfb//2), (4,4), strides=2, padding="same", name=f"{self.name}_final_up") # Ensure min 16 filters
        self.final_relu = Activation("relu", name=f"{self.name}_final_relu")
        self.final_conv_out = Conv2D(1, (3,3), activation='sigmoid', padding='same', name=f"{self.name}_final_out_conv")

    def call(self, inputs):
        planned_features, original_outline, style_tokens_for_bottleneck = inputs

        # --- Prepare multi-scale plan features and outlines ---
        plan_l0_128 = planned_features
        outline_l0_128 = original_outline
        
        plan_l1_64 = self.pool(plan_l0_128)
        outline_l1_64 = self.pool(outline_l0_128)
        plan_l2_32 = self.pool(plan_l1_64)
        outline_l2_32 = self.pool(outline_l1_64)
        plan_l3_16 = self.pool(plan_l2_32)
        outline_l3_16 = self.pool(outline_l2_32)

        # --- Encoder ---
        # Enc1 (Input: 128x128) -> Output enc1_output (64x64)
        x = concatenate([plan_l0_128, outline_l0_128], name=f"{self.name}_e1_in_concat")
        x = self.enc1_conv1(x); x = self.enc1_lrelu1(x)
        enc1_output = self.enc1_conv2_down(x); enc1_output = self.enc1_bn2(enc1_output); enc1_output = self.enc1_lrelu2(enc1_output)

        # Enc2 (Input: 64x64) -> Output enc2_output (32x32)
        x = concatenate([enc1_output, plan_l1_64, outline_l1_64], name=f"{self.name}_e2_in_concat")
        x = self.enc2_conv1(x); x = self.enc2_bn1(x); x = self.enc2_lrelu1(x)
        enc2_output = self.enc2_conv2_down(x); enc2_output = self.enc2_bn2(enc2_output); enc2_output = self.enc2_lrelu2(enc2_output)
        
        # Enc3 (Input: 32x32) -> Output enc3_output (16x16)
        x = concatenate([enc2_output, plan_l2_32, outline_l2_32], name=f"{self.name}_e3_in_concat") # This was the error line
        x = self.enc3_conv1(x); x = self.enc3_bn1(x); x = self.enc3_lrelu1(x)
        enc3_output = self.enc3_conv2_down(x); enc3_output = self.enc3_bn2(enc3_output); enc3_output = self.enc3_lrelu2(enc3_output)

        # Enc4 (Input: 16x16) -> Output enc4_features_for_bottleneck (8x8)
        x = concatenate([enc3_output, plan_l3_16, outline_l3_16], name=f"{self.name}_e4_in_concat")
        x = self.enc4_conv1(x); x = self.enc4_bn1(x); x = self.enc4_lrelu1(x)
        enc4_features_for_bottleneck = self.enc4_conv2_down(x); enc4_features_for_bottleneck = self.enc4_bn2(enc4_features_for_bottleneck); enc4_features_for_bottleneck = self.enc4_lrelu2(enc4_features_for_bottleneck)
        
        bottleneck_modulated = self.adaptive_bottleneck([enc4_features_for_bottleneck, style_tokens_for_bottleneck])

        # --- Decoder ---
        # Dec3 (Input: bottleneck_modulated 8x8, Skip: enc3_output 16x16, Plan: plan_l3_16 16x16) -> Output dec3_output (16x16)
        x = self.dec3_upconv(bottleneck_modulated); x = self.dec3_bn1(x); x = self.dec3_relu1(x)
        x = concatenate([x, enc3_output, plan_l3_16], name=f"{self.name}_d3_concat") # Use plan_l3_16
        x = self.dec3_conv1(x); x = self.dec3_bn2(x); x = self.dec3_relu2(x)
        dec3_output = self.dec3_conv2(x); dec3_output = self.dec3_bn3(dec3_output); dec3_output = self.dec3_relu3(dec3_output)

        # Dec2 (Input: dec3_output 16x16, Skip: enc2_output 32x32, Plan: plan_l2_32 32x32) -> Output dec2_output (32x32)
        x = self.dec2_upconv(dec3_output); x = self.dec2_bn1(x); x = self.dec2_relu1(x)
        x = concatenate([x, enc2_output, plan_l2_32], name=f"{self.name}_d2_concat")
        x = self.dec2_conv1(x); x = self.dec2_bn2(x); x = self.dec2_relu2(x)
        dec2_output = self.dec2_conv2(x); dec2_output = self.dec2_bn3(dec2_output); dec2_output = self.dec2_relu3(dec2_output)

        # Dec1 (Input: dec2_output 32x32, Skip: enc1_output 64x64, Plan: plan_l1_64 64x64) -> Output dec1_output (64x64)
        x = self.dec1_upconv(dec2_output); x = self.dec1_bn1(x); x = self.dec1_relu1(x)
        x = concatenate([x, enc1_output, plan_l1_64], name=f"{self.name}_d1_concat")
        x = self.dec1_conv1(x); x = self.dec1_bn2(x); x = self.dec1_relu2(x)
        dec1_output = self.dec1_conv2(x); dec1_output = self.dec1_bn3(dec1_output); dec1_output = self.dec1_relu3(dec1_output)

        # Final Output (Upsample to 128x128)
        x = self.final_up(dec1_output)
        x = self.final_relu(x)
        output_image = self.final_conv_out(x)
        
        return output_image
    
    def get_config(self): # For saving/loading Model subclass
        config = super().get_config()
        config.update({
            "initial_plan_channels": self.initial_plan_channels,
            "num_filters_base": self.nfb,
            "style_token_dim": self.adaptive_bottleneck.style_token_dim,
            "num_style_tokens_for_bottleneck": self.adaptive_bottleneck.num_style_tokens,
            "img_dim_generator": self.img_dim_generator
        })
        return config
    @classmethod
    def from_config(cls, config): # For saving/loading Model subclass
        return cls(**config)

# --- Full Hybrid Model ---
# (build_full_hybrid_model function and the if __name__ == '__main__': block remain the same as my previous message)
# ... (ensure it's updated if AdaptiveGenerator constructor changed, e.g., img_dim_generator)
def build_full_hybrid_model(
    img_shape=(IMG_DIM, IMG_DIM, 1), 
    num_examples=5,
    patch_size_style=PATCH_SIZE_STYLE, embed_dim_style=128, num_heads_style=4, 
    num_transformer_layers_style=2, num_style_tokens=5,
    plan_channels=64, num_attn_heads_plan=4,
    gen_filters_base=32,
    name="hybrid_cat_generator"
):
    example_img_inputs = [Input(shape=img_shape, name=f"example_img_input_{i}") for i in range(num_examples)]
    outline_input = Input(shape=img_shape, name="outline_main_input")

    style_encoder = StyleEncoder(
        num_examples=num_examples, patch_size=patch_size_style, img_dim_encoder=img_shape[0],
        embed_dim=embed_dim_style, num_heads=num_heads_style, 
        num_transformer_layers=num_transformer_layers_style, num_style_tokens=num_style_tokens
    )
    style_tokens = style_encoder(example_img_inputs)

    spatial_planner = SpatialPlanner(
        style_token_dim=embed_dim_style,
        plan_channels=plan_channels, 
        num_attn_heads=num_attn_heads_plan
    )
    planned_features = spatial_planner([style_tokens, outline_input])

    adaptive_generator = AdaptiveGenerator(
        initial_plan_channels=plan_channels,
        num_filters_base=gen_filters_base,
        style_token_dim=embed_dim_style,
        num_style_tokens_for_bottleneck=num_style_tokens,
        img_dim_generator=img_shape[0] # Pass image dimension
    )
    generated_output = adaptive_generator([planned_features, outline_input, style_tokens])

    model = Model(inputs=example_img_inputs + [outline_input], outputs=generated_output, name=name)
    return model

# --- Custom Loss: Distance-Weighted Soft Spatial Loss ---
DISTANCE_LOSS_SIGMA = 3.0 

@tf.keras.utils.register_keras_serializable(package="MyCustomLosses") # Decorator for serialization
class DistanceWeightedSoftSpatialLoss(Loss):
    def __init__(self, sigma=DISTANCE_LOSS_SIGMA, name="distance_weighted_soft_spatial_loss", **kwargs): # Added **kwargs
        super().__init__(name=name, **kwargs) # Pass **kwargs to base
        self.sigma = tf.constant(sigma, dtype=tf.float32)

    def call(self, y_true_and_outline_for_loss, y_pred_grayscale):
        target_grayscale_image = y_true_and_outline_for_loss[..., 0:1]
        outline_mask_object_is_1 = 1.0 - y_true_and_outline_for_loss[..., 1:2]

        def calculate_weights_py_fn(single_outline_mask_object_is_1_float32_np):
            # This function now receives a NumPy array
            # single_outline_mask_object_is_1_float32_np shape is (H, W, 1)
            mask_uint8_squeezed = (single_outline_mask_object_is_1_float32_np * 255.0).astype(np.uint8)[..., 0]
            
            # Distance from each pixel to the nearest boundary of the object region
            # Invert mask so boundary is between 0 and 255 regions
            # dist_to_boundary = cv2.distanceTransform(mask_uint8_squeezed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) # dist inside object
            # dist_to_boundary_outside = cv2.distanceTransform(cv2.bitwise_not(mask_uint8_squeezed), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) # dist outside object
            # combined_dist = np.where(mask_uint8_squeezed > 0, dist_to_boundary, -dist_to_boundary_outside) # Negative outside

            # Simpler: Let's use the "full loss in interior, weighted near boundaries"
            # The `mask_uint8` (object=255, bg=0) is a good starting point for "interior"
            # The "softness" will come more from approximate outlines than complex weighting for now.
            # For now, let's just use the object_region_mask as the weight.
            # The paper suggested exp(-distance/sigma) "near boundaries".
            # This means pixels *at* the boundary have weight exp(0)=1.
            # Pixels *just outside* (dist=1) have weight exp(-1/sigma).
            # Pixels *just inside* (dist=1 from other side) have weight exp(-1/sigma).
            # This is complex to implement perfectly with current distance transform.

            # Let's use a simplified approach: high weight inside, some weight at boundary, low outside.
            # For this version, we will use the object_region_mask itself (object=1, bg=0) as the primary loss weight.
            # The use of approximate outlines during training is the main driver for "softness".
            weights = mask_uint8_squeezed.astype(np.float32) / 255.0 
            return np.expand_dims(weights, axis=-1) # Return as (H, W, 1)

        loss_weights = tf.py_function(
            func=calculate_weights_py_fn, 
            inp=[outline_mask_object_is_1], # Pass the tensor directly
            Tout=tf.float32
        )
        loss_weights.set_shape([None, IMG_DIM, IMG_DIM, 1])

        pixel_wise_mse = tf.square(target_grayscale_image - y_pred_grayscale)
        weighted_pixel_wise_mse = pixel_wise_mse * loss_weights
        
        sum_weighted_mse = tf.reduce_sum(weighted_pixel_wise_mse, axis=[1,2,3])
        sum_weights = tf.reduce_sum(loss_weights, axis=[1,2,3]) + 1e-7
        
        mean_loss_per_sample = sum_weighted_mse / sum_weights
        return tf.reduce_mean(mean_loss_per_sample)

    def get_config(self):
        config = super().get_config()
        config.update({"sigma": self.sigma.numpy() if tf.is_tensor(self.sigma) else self.sigma})
        return config
    # from_config is often handled by base Loss class if get_config is sufficient
    # and __init__ takes standard args + serializable custom args.

if __name__ == '__main__':
    print(f"TensorFlow Version: {tf.__version__}")
    model = build_full_hybrid_model(
        embed_dim_style=128, 
        num_transformer_layers_style=2,
        plan_channels=64, 
        gen_filters_base=24 
    )
    model.summary(line_length=180)
    print("\nAttempting to save and reload model for serializability check...")
    temp_model_path = "temp_hybrid_model_check.keras"
    custom_objects_for_test = {
        'FiLMLayer': FiLMLayer, 'PatchEmbed': PatchEmbed, 'StyleEncoder': StyleEncoder,
        'SpatialPlanner': SpatialPlanner, 'AdaptiveBottleneck': AdaptiveBottleneck,
        'AdaptiveGenerator': AdaptiveGenerator, 'DistanceWeightedSoftSpatialLoss': DistanceWeightedSoftSpatialLoss
    }
    try:
        model.save(temp_model_path)
        loaded_model = tf.keras.models.load_model(temp_model_path, custom_objects=custom_objects_for_test)
        print(f"Successfully saved and loaded: {temp_model_path}")
        if os.path.exists(temp_model_path): os.remove(temp_model_path)
        print("SERIALIZABILITY CHECK PASSED!")
        print("\nTesting model with dummy data...")
        batch_size_test = 1 
        dummy_examples = [np.random.rand(batch_size_test, IMG_DIM, IMG_DIM, 1).astype(np.float32) for _ in range(5)]
        dummy_outline = np.random.rand(batch_size_test, IMG_DIM, IMG_DIM, 1).astype(np.float32)
        dummy_output = model.predict(dummy_examples + [dummy_outline])
        print(f"Dummy output shape: {dummy_output.shape}")
    except Exception as e:
        print(f"SERIALIZABILITY CHECK OR DUMMY PREDICTION FAILED: {e}")
        traceback.print_exc()
    print("-" * 50)
