import  numpy as np
import  cv2
from os import listdir

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


X_positive = load_images(1, 1, "C://Users/steffen/LuftbildarchiologiePositiv/")
X_negative = load_images(1, 1, "C://Users/steffen/LuftbildarchiologieNegativ/")
print(X_positive.shape)
print(X_negative.shape)
All_X = np.concatenate((X_positive, X_negative), axis=0)

Y_train_true = np.ones(624, dtype=bool)
Y_train_false = np.zeros(3959, dtype=bool)

Y_train_all = np.concatenate((Y_train_true, Y_train_false), axis=0)
print(Y_train_all)

print(All_X.shape)
print(All_X)