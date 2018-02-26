from keras.models import Model
from keras.layers import Conv2D, Input, Conv2DTranspose, merge
from keras.optimizers import SGD, adam
from keras.callbacks import ModelCheckpoint
#import prepare_data as pd
import pandas
import numpy
import cv2

scale = 2

def model_EES16():
    _input = Input(shape=(None, None, 1), name='input')

    EES = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    EES = Conv2DTranspose(filters=32, kernel_size=(14, 14), strides=(2, 2), padding='same', activation='relu')(EES)
    out = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(EES)

    model = Model(input=_input, output=out)

    return model


def EES_train():
    EES = model_EES16()
    EES.compile(optimizer=adam(lr=0.0003), loss='mse')
    print EES.summary()
    data, label = pd.read_training_data("./lil_train.h5")
    val_data, val_label = pd.read_training_data("./little_test.h5")

    checkpoint = ModelCheckpoint("EES_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    history_callback = EES.fit(data, label, batch_size=64, validation_data=(val_data, val_label),
                               callbacks=callbacks_list, shuffle=True, nb_epoch=100, verbose=1)
    EES.save_weights("EES_final.h5")


def EES_predict():
    IMG_NAME = "./input.jpg"
    INPUT_NAME = "./downscaled.jpg"
    OUTPUT_NAME = "./output.jpg"

    label = cv2.imread(IMG_NAME)
    shape = label.shape

    img = cv2.resize(label, (shape[1] / scale, shape[0] / scale), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    EES = model_EES16()
    EES.load_weights("EES_check.h5")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
    img = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

    pre = EES.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre = numpy.uint8(pre)
    img[:, :, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
    #cv2.imwrite("Bicubic.jpg", cv2.cvtColor(im2, cv2.COLOR_YCrCb2BGR))
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

  #  print "Bicubic:"
   # print cv2.PSNR(im1[:, :, 0], im2[:, :, 0])
    print "EES:"
    print cv2.PSNR(im1[:, :, 0], im3[:, :, 0])



if __name__ == "__main__":
    #EES_train()
    EES_predict()
