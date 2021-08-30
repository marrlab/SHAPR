from shapeae.utils import *
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from shapeae import settings
from shapeae.data_generator import *
from shapeae.model import model, define_adverserial


PARAMS = {"num_filters": 10,
      "dropout": 0.
}

"""
Set the path where the following folders are located: 
- obj: containing the 3D groundtruth segmentations 
- mask: containg the 2D masks 
- image: containing the images from which the 2D masks were segmented (e.g. brightfield)
All input data is expected to have the same x and y dimensions and the obj (3D segmentations to have a z-dimension of 64.
The filenames of corresponding files in the obj, mask and image ordner are expeted to match.
"""


def run_train():

    print(settings)
    """
    Get the filenames
    """
    filenames = os.listdir(os.path.join(settings.path, "obj"))

    """
    We train the model on all data on 5 folds, while the folds are randomly split
    """
    kf = KFold(n_splits=5)
    os.makedirs(os.path.join(settings.path, "logs"), exist_ok=True)

    for fold, (cv_train_indices, cv_test_indices) in enumerate(kf.split(filenames)):
        cv_train_filenames = [str(filenames[i]) for i in cv_train_indices]
        cv_test_filenames = [str(filenames[i]) for i in cv_test_indices]

        """
        From the train set we use 20% of the files as validation during training 
        """
        train_filenames, val_filenames = train_test_split(
            cv_train_filenames, 
            test_size=0.2,  
            random_state = settings.random_seed
        )

        print(f"For Validation we use: {str(len(val_filenames))} randomly sampled files")
        print(f"For training we use: {str(len(train_filenames))} randomly sampled files")
        ShapeAEmodel, discriminator = model(PARAMS)

        """
        If pretrained weights should be used, please add them here:
        These weights will be used for all folds
        """
        pretrained_weightsPredictor_file = os.path.join(settings.path, "logs", "pretrained_weightsPredictor"+str(fold) +".hdf5")

        model_checkpoint = ModelCheckpoint(pretrained_weightsPredictor_file, monitor=('val_loss'), verbose=1, save_best_only=True)
        es_callback = EarlyStopping(monitor='val_loss', patience=15)
        tensorboard_callback = TensorBoard(log_dir= os.path.join(settings.path, "logs", format(time.time())))

        print("the number of validation files is:", len(val_filenames))
        print("the number of training files is:", len(train_filenames))
        val_data = data_generator(settings.path, val_filenames, settings.batch_size)
        train_data = data_generator(settings.path, train_filenames, settings.batch_size)

        ShapeAEmodel.fit_generator(
            train_data,
            epochs = settings.epochs_ShapeAE,
            steps_per_epoch = int(len(train_filenames)/settings.batch_size),
            validation_data = val_data,
            validation_steps=len(val_filenames),
            callbacks=[model_checkpoint, es_callback, tensorboard_callback]
        )
        ShapeAEmodel.load_weights(pretrained_weightsPredictor_file)
        """
        After training ShapeAE for the set number of epochs, we train the adverserial model
        """

        def train(ShapeAEmodel, discriminator, adverserialmodel, train_filenames, pretrained_weightsPredictor_GAN,
                  n_epochs=50, batch_size=6):
            # manually enumerate epochs
            gan_losses = []
            disc_losses = []
            shapeAE_loss = []
            validation_error = []
            best_val_error = get_validation_error(ShapeAEmodel, settings.path, val_filenames)
            print("validation error", best_val_error)
            validation_error.append(best_val_error)
            for i in range(n_epochs):
                # enumerate batches over the training set
                for j in range(int(len(train_filenames) / batch_size)):
                    train_filename = list(train_filenames[j * batch_size:j * batch_size + batch_size])
                    # get randomly selected 'real' samples
                    nr_train_filename = int(len(train_filename) / 2)
                    X_real, y_real = generate_real_sample(settings.path, train_filename[0:nr_train_filename], nr_train_filename)
                    # generate 'fake' examples
                    X_fake, y_fake = generate_fake_sample(ShapeAEmodel, settings.path, train_filename[nr_train_filename:],
                                                          nr_train_filename)
                    # create training set for the discriminator
                    X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
                    # train the discriminator
                    discriminator.trainable = True
                    d_loss, _ = discriminator.train_on_batch(X, y)
                    discriminator.trainable = False
                    # train the generator
                    y_gan = np.ones((batch_size, 1))
                    x_gan, y_generator = data_generator_adserial(settings.path, train_filename, batch_size)
                    g_loss = adverserialmodel.train_on_batch(x_gan, [y_generator, y_gan])
                    # save loss functions
                    gan_losses.append(g_loss[0])
                    disc_losses.append(d_loss)
                    # summarize loss
                    if (j + 1) % (10) == 0:
                        print("epoch", i + 1, "of", settings.epochs_cShapeAE, "step", j + 1, "of",
                              int(len(train_filenames) / batch_size),
                              "// ShapeAE adverserial loss:", np.mean(gan_losses[-10:]),
                              "// discriminator loss", np.mean(disc_losses[-10:]))
                        print("gan loss", g_loss)
                print("finished epoch", i + 1)
                # evaluted model on validation set
                val_error = get_validation_error(ShapeAEmodel, settings.path, val_filenames)
                print("validation error", val_error)
                validation_error.append(val_error)
                # save model only if it has improved
                if val_error < best_val_error:
                    print("val_loss has improved from", best_val_error, "to", val_error)
                    best_val_error = val_error
                    ShapeAEmodel.save_weights(pretrained_weightsPredictor_GAN)

        pretrained_weightsPredictor_GAN = os.path.join(settings.path, "logs", "pretrained_weightsPredictor_GAN" + str(fold) + ".hdf5")
        adverserialmodel = define_adverserial(ShapeAEmodel, discriminator)
        train(ShapeAEmodel, discriminator, adverserialmodel, train_filenames, pretrained_weightsPredictor_GAN,
              n_epochs=settings.epochs_cShapeAE, batch_size=settings.batch_size)

        """
        The 3D shape of the test data for each fold will be predicted here
        """

        test_data = data_generator_test_set(settings.path, cv_test_filenames)

        predict = ShapeAEmodel.predict_generator(test_data, steps = len(cv_test_filenames))
        print(np.shape(predict))

        """
        The predictions on the test set for each fold will be saved to the results folder
        """
        #save predictions
        print(np.shape(predict))
        for i, test_filename in enumerate(cv_test_filenames):
            result = predict[i,...]*255
            os.makedirs(settings.result_path, exist_ok=True)
            imsave(os.path.join(settings.result_path, test_filename), result.astype("uint8"))
            i += 1


def run_evaluation(): 

    print(settings)

    """
    Get the filenames
    """
    test_filenames = os.listdir(os.path.join(settings.path, "obj"))

    model2D = model(PARAMS)
    model2D.load_weights(settings.pretrained_weights_path)

    """
    If pretrained weights should be used, please add them here:
    These weights will be used for all folds
    """

    """
    The 3D shape of the test data for each fold will be predicted here
    """
    test_data = data_generator_test_set(settings.path, test_filenames)

    predict = model2D.predict_generator(test_data, steps = len(test_filenames))
    print(np.shape(predict))

    """
    The predictions on the test set for each fold will be saved to the results folder
    """
    #save predictions
    print(np.shape(predict))
    i = 0
    for i, test_filename in enumerate(test_filenames):
        result = predict[i,...]*255
        os.makedirs(settings.result_path, exist_ok=True)
        imsave(settings.result_path + test_filename, result.astype("uint8"))
        i = i+1            


