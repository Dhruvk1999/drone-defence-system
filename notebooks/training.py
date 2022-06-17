# all the required lib
import os, cv2, keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
#from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


print("creating Selective Segmentation object")
# selective segmentation
ss= cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

path = "Images"
annot = "Airplanes_Annotations"


train_images=[]
train_labels=[]


for i in range(456,600):
    try:
        
            
        filename = f"{i}.JPG"
        print(filename)

       
        image = cv2.imread(f"../data/drone/{filename}")

        df = pd.read_csv(f"../data/drone/annot/{i}.csv")

        gtvalues=[]
        for row in df.iterrows():

            #change these maybe
            x1 = int(row[1][0].split(" ")[0])
            y1 = int(row[1][0].split(" ")[1])
            x2 = int(row[1][0].split(" ")[2])
            y2 = int(row[1][0].split(" ")[3])
            gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = image.copy()
        counter = 0
        falsecounter = 0
        flag = 0
        fflag = 0
        bflag = 0
        for e,result in enumerate(ssresults):
            if e < 2000 and flag == 0:
                for gtval in gtvalues:
                    x,y,w,h = result
                    iou = get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                    if counter < 30:
                        if iou > 0.70:
                            timage = imout[y:y+h,x:x+w]
                            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(1)
                            counter += 1
                    else :
                        fflag =1
                    if falsecounter <30:
                        if iou < 0.3:
                            timage = imout[y:y+h,x:x+w]
                            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(0)
                            falsecounter += 1
                    else :
                        bflag = 1
                if fflag == 1 and bflag == 1:
                    print("inside")
                    flag = 1
    except Exception as e:
        print(e)
        print("error in "+filename)
        continue
        
X_new = np.array(train_images)
y_new = np.array(train_labels)

print(f"the shape of x = {X_new.shape}")


print("downloading the model .... ")
vggmodel = VGG16(weights='imagenet', include_top=True)
#vggmodel.summary()

print("\n download complete")


for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False

    
# not sure what these lines do
X= vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(input = vggmodel.input, output = predictions)
opt = Adam(lr=0.0001)
model_final.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_final.summary()


# calling the mybinarizer object
# to-do 
# add import for mylabelbinarizer and iou func

#train-test split
X_train, X_test , y_train, y_test = train_test_split(X_new,Y,test_size=0.10)

print("shape of train test split : \n")
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# data augmentation

trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)


# trained model
hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 10, epochs= 1000, validation_data= testdata, validation_steps=2, callbacks=[checkpoint,early])

print("the model is trained")

# saving the weights and architecture
model_final("weights.h5")
model_final.save('architecure.h5')



