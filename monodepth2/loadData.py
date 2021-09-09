import  numpy as np
import  cv2
from os import listdir
from keras.preprocessing.image import ImageDataGenerator

def load_images(wide, height, path):

    array = np.empty([0, wide, height, 3])

    names = [p for p in listdir(path)if p[-4:] == ".jpg"]

    for i in names:
        #print(path + i)
        img = cv2.imread(path + i) / 255.
        img = cv2.resize(img, (wide, height), interpolation = cv2.INTER_AREA)
        img = np.expand_dims(img, 0)
        array = np.concatenate((array, img), axis=0)
        print(i)
    return array


X_positive = load_images(128, 128, "C://Users/steffen/LuftbildarchiologiePositiv/")
X_negative = load_images(128, 128, "C://Users/steffen/LuftbildarchiologieNegativ/")
print(X_positive.shape)
print(X_negative.shape)
All_X = np.concatenate((X_positive, X_negative), axis=0)

Y_train_true = np.ones(624, dtype=bool)
Y_train_false = np.zeros(3959, dtype=bool)

Y_train_all = np.concatenate((Y_train_true, Y_train_false), axis=0)
print(Y_train_all)

print(All_X.shape)
print(All_X)

gen = ImageDataGenerator(width_shift_range=3,
                         height_shift_range=3,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         vertical_flip=True
                         )
"""
print(gen.flow(All_X, Y_train_all, shuffle=True, batch_size=4583))

print(Y_train_all)
print(Y_train_all.shape)

print(All_X.shape)
print(All_X)
"""
np.save("C://Users/steffen/repos/tello/monodepth2/X_train_non_gen.npy", All_X, allow_pickle=False)
np.save("C://Users/steffen/repos/tello/monodepth2/Y_train_non_gen.npy", Y_train_all, allow_pickle=False)
