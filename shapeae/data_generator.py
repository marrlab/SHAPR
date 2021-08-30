from ._settings import settings
from .utils import *
from .metrics import *

def augmentation(obj, img):
    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)
    if random.choice([True, True, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 1)
        img = np.flip(img, len(np.shape(img)) - 1)
    if random.choice([True, True, False]) == True:
        obj = np.flip(obj, len(np.shape(obj)) - 2)
        img = np.flip(img, len(np.shape(img)) - 2)

    if random.choice([True, True, False]) == True:
        angle = np.random.choice(int(360 * 100)) / 100
        img = np.nan_to_num(rotate(img, angle, resize=False, preserve_range=True))
        for i in range(0, np.shape(obj)[0]):
            obj[i, :, :] = np.nan_to_num(rotate(obj[i, :, :], angle, resize=False, preserve_range=True))

    if random.choice([True, True, False]) == True:
        from skimage.util import random_noise
        img = random_noise(img, mode='gaussian', var= 0.02)

    if random.choice([True, True, False]) == True:
        obj_shape = np.shape(obj)
        img_shape = np.shape(img)
        x_shift = np.random.choice(int(40))
        y_shift = np.random.choice(int(40))
        x_shift2 = np.random.choice(int(40))
        y_shift2 = np.random.choice(int(40))
        z_shift = np.random.choice(int(10))
        z_shift2 = np.random.choice(int(10))
        obj = obj[z_shift:-(z_shift2+1), x_shift:-(x_shift2+1), y_shift:-(y_shift2+1)]
        img = img[int(x_shift/4):-int(x_shift2/4+1), int(y_shift/4):-int(y_shift2/4+1),:]
        obj = resize(obj, obj_shape, preserve_range=True)
        img = resize(img, img_shape, preserve_range=True)

    return obj, img



"""
The data generator will open the 3D segmentation, 2D masks and 2D images for each fold from the directory given the filenames and return a tensor
The 2D mask and the 2D image will be multiplied pixel-wise to remove the background
"""
def data_generator(path, filenames, batch_size):
    while True:
        def grouped(filenames, batch_size):
            return zip(*[iter(filenames)] * batch_size)
        for train_files in grouped(filenames, batch_size):
            obj_out = []
            mask_bf_out = []
            for train_file in train_files:
                if not train_file.startswith('.'):
                    obj = import_image(os.path.join(path, "obj", train_file)) / 255.
                    img = import_image(os.path.join(path, "mask", train_file)) / 255.
                    bf = import_image(os.path.join(path, "image", train_file)) / 255.
                    mask_bf = np.zeros((int(np.shape(img)[0]), int(np.shape(img)[1]), 2))
                    mask_bf[:, :, 0] = img
                    mask_bf[:, :, 1] = bf * img
                    #obj, mask_bf = augmentation(obj, mask_bf)
                    obj_out.append(obj[:,:,:,np.newaxis])
                    mask_bf_out.append(mask_bf[np.newaxis,...])
            X_batch = np.array(mask_bf_out)
            Y_batch = np.array(obj_out)
            yield X_batch, Y_batch

def data_generator_adserial(path, filenames, batch_size):
    #while True:
    def grouped(filenames, batch_size):
        return zip(*[iter(filenames)] * batch_size)
    for train_files in grouped(filenames, batch_size):
        obj_out = []
        mask_bf_out = []
        for train_file in train_files:
            if not train_file.startswith('.'):
                obj = import_image(os.path.join(path, "obj", train_file)) / 255.
                img = import_image(os.path.join(path, "mask", train_file)) / 255.
                bf = import_image(os.path.join(path, "image", train_file)) / 255.
                mask_bf = np.zeros((int(np.shape(img)[0]), int(np.shape(img)[1]), 2))
                mask_bf[:, :, 0] = img
                mask_bf[:, :, 1] = bf * img
                #obj, mask_bf = augmentation(obj, mask_bf)
                obj_out.append(obj[:,:,:,np.newaxis])
                mask_bf_out.append(mask_bf[np.newaxis,...])
        X_batch = np.array(mask_bf_out)
        Y_batch = np.array(obj_out)
        return X_batch, Y_batch


"""
The test data generator will open the 2D masks and 2D images for each fold from the directory given the filenames and return a tensor
"""
def data_generator_test_set(path, filenames):
    while True:
        for test_file in filenames:
            if not test_file.startswith('.'):
                img = import_image(os.path.join(path, "mask", test_file)) / 255.
                bf = import_image(os.path.join(path, "image", test_file)) /255.
                img_out_2 = np.zeros((int(np.shape(img)[0]), int(np.shape(img)[1]), 2))
                img_out_2[:, :, 0] = img
                img_out_2[:, :, 1] = bf * img
                img_out_2 = img_out_2[np.newaxis, np.newaxis, ...]
                yield img_out_2

def generate_real_sample(path, train_filename, batch_size):
    _, real_sample = data_generator_adserial(path, train_filename, batch_size)
    y = np.ones((batch_size, 1))
    return real_sample, y

def generate_fake_sample(ShapeAEmodel, path, train_filename, batch_size):
    example_2D,_ = data_generator_adserial(path, train_filename, batch_size)
    y = np.zeros((batch_size, 1))
    fake_sample = ShapeAEmodel.predict(example_2D)
    fake_sample = np.array(fake_sample)
    return fake_sample, y

def get_validation_error(ShapeAEmodel, path, val_filenames):
    x_val, y_true = data_generator_adserial(path, val_filenames, 4)
    y_pred = ShapeAEmodel.predict(x_val)
    # val_error = mean_squared_error(np.array(y_true.flatten()), np.array(y_pred.flatten()))
    val_error = dice_crossentropy_loss(y_true.astype("float"), y_pred.astype("float"))
    return np.mean(val_error)