from tensorflow import keras
import tensorflow as tf

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Flatten

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

ANIMALS_PATH = "C:/Users/Hareesh/Downloads/archive (4)/animals/animals"
IMAGE_SIZE = [224, 224]
ANIMAL_TYPES = 90
BATCH_SIZE = 30
EPOCHS = 15

AnimalModel = VGG16(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# freeze layers of predefined model.
for layer in AnimalModel.layers:  
    layer.trainable = False
    
# add a flatenning layer and output layer.
FlattenedLayer = Flatten()(AnimalModel.output)
OutputLayer = Dense(ANIMAL_TYPES, activation='softmax')(FlattenedLayer)

AnimalModel = Model(inputs=AnimalModel.input, outputs=OutputLayer)

AnimalModel.compile(
                      loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy']
                    )
ImageGen = ImageDataGenerator(rescale=1./255, validation_split = 0.2)

TrainGen = ImageGen.flow_from_directory(
                                                    directory=ANIMALS_PATH,
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    subset='training',
                                                    interpolation='bicubic',
                                        )
TestGen  = ImageGen.flow_from_directory(
                                                    directory=ANIMALS_PATH,
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    subset='validation',
                                                    interpolation='bicubic',
                                        )

stats = AnimalModel.fit_generator(
                                        generator = TrainGen,
                                        validation_data = TestGen,
                                        epochs = EPOCHS,
                                        steps_per_epoch= len(TrainGen.filenames)//BATCH_SIZE,
                                        validation_steps=len(TestGen.filenames)//BATCH_SIZE
                                  )
AnimalModel.save("AnimalRecognizer.h5")


img = image.load_img("C:/Users/Hareesh/Downloads/archive (4)/animals/animals/wombat/6f69d6f98f.jpg", target_size=IMAGE_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
pred = np.argmax(AnimalModel.predict(x))
out = ['antelope','badger','bat','bear','bee','beetle','bison','boar','butterfly','cat','caterpillar','chimpanzee','cockroach','cow','coyote','crab','crow','deer','dog','dolphin','donkey','dragonfly','duck',
'eagle',
'elephant',
'flamingo',
'fly',
'fox',
'goat',
'goldfish',
'goose',
'gorilla',
'grasshopper',
'hamster',
'hare',
'hedgehog',
'hippopotamus',
'hornbill',
'horse',
'hummingbird',
'hyena',
'jellyfish',
'kangaroo',
'koala',
'ladybugs',
'leopard',
'lion',
'lizard',
'lobster',
'mosquito',
'moth',
'mouse',
'octopus',
'okapi',
'orangutan',
'otter',
'owl',
'ox',
'oyster',
'panda',
'parrot',
'pelecaniformes',
'penguin',
'pig',
'pigeon',
'porcupine',
'possum',
'raccoon',
'rat',
'reindeer',
'rhinoceros',
'sandpiper',
'seahorse',
'seal',
'shark',
'sheep',
'snake',
'sparrow',
'squid',
'squirrel',
'starfish',
'swan',
'tiger',
'turkey',
'turtle',
'whale',
'wolf',
'wombat',
'woodpecker',
'zebra']

print(out[pred])

# Given picture is a worm bat and the model predicted it correct
#The AnimalModel has an accuracy of 89%