import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.layers import MultiHeadAttention


class PatchCreationLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class ViTLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, num_layers, num_heads, mlp_dim, patch_size):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = 64
        self.mlp_dim = mlp_dim
        self.embed_dim = self.num_heads * self.head_size
        self.patch_size = patch_size
        self.patcher = PatchCreationLayer(self.patch_size)
        self.transformer_encoder = [
            tf.keras.Sequential(
                [
                    MultiHeadAttention(num_heads=self.num_heads,
                                       key_dim=self.head_size),
                    Dropout(0.1),
                    LayerNormalization(),
                    Dense(units=self.mlp_dim, activation=tf.nn.gelu),
                    Dropout(0.1),
                    LayerNormalization(),
                ]
            )
            for _ in range(self.num_layers)
        ]
        self.classifier = Dense(num_classes)

    def call(self, images):
        default_batch_size = 32
        batch_size = tf.shape(images)[0]
        patches = self.patcher(images)
        if patches.shape[0] is None:
            patches.set_shape([default_batch_size] + patches.shape[1:])
        positions = tf.Variable(
            initial_value=tf.zeros(shape=(1, patches.shape[1], self.embed_dim)),
            trainable=True,
        )
        x = patches + positions
        for layer in self.transformer_encoder:
            x = layer(x)
        x = self.classifier(tf.reduce_mean(x, axis=1))
        return x


def create_vit_classifier():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    resizing = Resizing(224, 224, interpolation="bilinear")(inputs)
    outputs = ViTLayer(
        num_classes=2,
        num_layers=2,
        num_heads=2,
        mlp_dim=128,
        patch_size=8,
    )(resizing)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


vit_classifier = create_vit_classifier()
vit_classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy")