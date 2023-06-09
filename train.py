import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from  utility import *
from   deeplearingmodel import  model


if __name__ == "__main__":
    test_data_path = 'test'
    train_data_path = 'train'
    val_data_path = 'val'
    batch_size = 64
    train_generator, val_generator, test_generator = create_generators(batch_size ,train_data_path,val_data_path,test_data_path)
    nbr_classes = train_generator.num_classes
    Train = True
    if Train:
        path_to_save_model = "./model"
        chk_saver = ModelCheckpoint(path_to_save_model,
                                     monitor  = 'val_accuracy',
                                     mode = 'max',
                                     save_best_only = True,
                                    save_freq = 'epoch',
                                      verbose = 1  )

        model = model(nbr_classes)
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=0.01)

        model.compile(optimizer ='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
        model.fit(train_generator,
                  epochs = 100,
                  batch_size=batch_size,
                  validation_data = val_generator,
                  callbacks= [chk_saver] )

    Test = True
    if Test:
        model =tf.keras.models.load_model('./model')
        model.summary()
        print("evaluating Validation set:")
        model.evaluate(val_generator)
        print("evaluating test set : ")
        model.evaluate(test_generator)







