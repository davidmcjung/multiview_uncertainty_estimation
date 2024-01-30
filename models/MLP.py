import tensorflow as tf
from tensorflow.keras import layers

class MLP(layers.Layer):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        is_bias=True,
        dropout=0.0,
        is_res=False,
        is_bn=False,
        l2=0.0,
        is_out_bn=False,
        out_activation=None,
        norm_type=None
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res
        self.is_bn = is_bn
        self.norm_type = norm_type
        if self.norm_type == 'layer_norm':
            norm = tf.keras.layers.LayerNormalization
        elif self.norm_type == 'batch_norm':
            norm = tf.keras.layers.BatchNormalization
        else:
            norm = tf.keras.layers.BatchNormalization
        self.dropout =  layers.Dropout(dropout) if dropout > 0 else tf.identity

   
        self.to_hidden = layers.Dense(
            self.hidden_size, 
            activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            use_bias=is_bias,
            input_shape=(self.input_size,),
            kernel_regularizer=tf.keras.regularizers.L2(l2), 
            )

        self.in_norm = norm()

        self.linears = [
            layers.Dense(
                self.hidden_size, 
                activation=tf.keras.layers.LeakyReLU(alpha=0.3), 
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                use_bias=is_bias,
                kernel_regularizer=tf.keras.regularizers.L2(l2), 
                )
            for _ in range(self.n_hidden_layers - 1)
            ]
        
        self.hidden_norms = [norm() for _ in range(self.n_hidden_layers - 1)]

        self.out = layers.Dense(
            self.output_size, 
            activation=out_activation, 
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            use_bias=is_bias,
            kernel_regularizer=tf.keras.regularizers.L2(l2), 
            )

        self.out_norm = norm()
        self.is_out_bn = is_out_bn


    def call(self, x, training=True):
        out = self.to_hidden(x)
        x = self.dropout(out)
        if self.is_bn:
            x = self.in_norm(x, training=training)

        for i,linear in enumerate(self.linears):
            out = linear(x)
            if self.is_res:
                out = out + x
            out = self.dropout(out)
            x = out
            if self.is_bn:
                x = self.hidden_norms[i](x, training=training)

        out = self.out(x)
        if self.is_out_bn:
            out = self.out_norm(out, training=training)
        return out