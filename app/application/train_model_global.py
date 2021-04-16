import app.config as cf
import os
from app.infrastructure.CleanData import CleanData
from app.model.DCNN import DCNN
from app.application.MyCustomCallback import MyCustomCallback
from app.application.pickling import to_pickle, from_pickle

import tensorflow as tf
from joblib import dump, load

tf.keras.backend.clear_session()


def train_model_global() :
    
    cd = CleanData(
        path=cf.INPUTS_FILE, 
        cols=cf.COLS,
        cols_to_keep=cf.COLS_TO_KEEP
    )

    tokenizer = cd.get_tokenizer()
    test_dataset, train_dataset = cd.get_train_test_dataset()

    
    VOCAB_SIZE = len(tokenizer.vocab)
    

    Dcnn = DCNNBERTEmbedding(
        nb_filters=NB_FILTERS,
        FFN_units=FFN_UNITS,
        nb_classes=NB_CLASSES,
        dropout_rate=DROPOUT_RATE
    )


    if NB_CLASSES == 2:
        Dcnn.compile(loss="binary_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])
    else:
        Dcnn.compile(loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=["sparse_categorical_accuracy"])

    
    ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cf.CHECKPOINT_PATH, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored!!")


    myCustomCallback = MyCustomCallback(ckpt_manager, cf.CHECKPOINT_PATH)


    Dcnn.fit(train_dataset, epochs=cf.NB_EPOCHS, callbacks=[myCustomCallback])


    results = Dcnn.evaluate(test_dataset)
    print(results)

    
    #to_pickle('Dcnn', Dcnn)
    
    
    #dump(Dcnn, os.path.join(cf.OUTPUTS_MODELS_DIR, 'Dcnn.joblib'))




if __name__ == "__main__":
    
    train_model_global()
