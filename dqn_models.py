from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Concatenate, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam

def create_model(size, lr = 0.01):
    s =  size
    input_me = Input(name="input-me", shape=(s, s, 1))
    input_op = Input(name="input-opponent", shape=(s, s, 1))
    # convolution_1 = Conv2D(14, kernel_size=2, strides=2, padding='valid', use_bias=False)
    convolution_1 = Conv2D(14, kernel_size=2, strides=2, padding='valid', activation='relu')
    # convolution_2 = Conv2D(32, kernel_size=2, activation='relu')

    conv_me = Flatten()(convolution_1(input_me))
    conv_op = Flatten()(convolution_1(input_op))
    # conv_me = Flatten()(convolution_2(convolution_1(input_me)))
    # conv_op = Flatten()(convolution_2(convolution_1(input_op)))

    concat_1 = Concatenate()([conv_me, conv_op])
    # dense_1 = Activation('tanh')(BatchNormalization()(Dense(64)(concat)))
    # dense_1 = Dense(128, activation='tanh')(concat_1)
    # dense_1 = Dense(128, activation='tanh')(concat_1)
    dense_1 = Dense(512, activation='relu')(concat_1)
    dense_2 = Dense(256, activation='relu')(dense_1)
    concat_2 = Concatenate()([Flatten()(input_me), Flatten()(input_op), dense_2])
    out = Dense(s ** 2, activation='linear')(concat_2)

    model = Model([input_me, input_op], out)
    model.compile(loss="mean_squared_error",
                  optimizer=Adam(lr=lr))
    return model