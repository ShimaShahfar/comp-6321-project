import numpy as np
import os
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import pydoc
from tensorflow.keras.preprocessing.image import Iterator
import matplotlib.pylab as plt


class CIHPDataGenerator(Iterator):
    # Generates data for Keras
    def __init__(
        self,
        source_dir,
        groundtruth_dir,
        target_shape,
        num_classes,
        batch_size,
        augmentation=False,
        shuffle=True,
        seed=None,
        **kwargs,
    ):

        # Initialization
        self.source_dir = source_dir
        self.groundtruth_dir = groundtruth_dir
        self.target_shape = tuple(target_shape)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augmentation = augmentation

        self.test = groundtruth_dir in ["", None]
        self.source_pairs = get_pairs_from_paths(
            source_dir, groundtruth_dir, test_time=self.test
        )

        super().__init__(len(self.source_pairs), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # Generates data containing batch_size samples
        images = [cv2.imread(self.source_pairs[i][0]) for i in index_array]

        if self.test:
            if self.augmentation:
                images = self.augmentation_sequence()(images=images)

            images_batch = self.process_images_batch(images)
            return images_batch

        # Load segmentations
        segmaps = [
            cv2.imread(self.source_pairs[i][1], flags=cv2.IMREAD_GRAYSCALE)[
                ..., np.newaxis
            ]
            for i in index_array
        ]

        # Do augmentations
        if self.augmentation:
            images, segmaps = self.augmentation_sequence()(
                images=images, segmentation_maps=segmaps
            )

        images_batch = self.process_images_batch(images)
        segmaps_batch = self.process_segmap_batch(segmaps)

        return images_batch, segmaps_batch

    def process_images_batch(self, images):
        return np.array(
            [cv2.resize(img, self.target_shape[::-1]) / 255 for img in images]
        )

    def process_segmap_batch(self, segmaps):
        segmaps_batch = np.array(
            [
                cv2.resize(
                    seg, self.target_shape[::-1], interpolation=cv2.INTER_NEAREST
                )
                for seg in segmaps
            ]
        )
        labels = np.arange(0, self.num_classes)
        onehot = segmaps_batch[..., np.newaxis] == labels
        return onehot.astype("float32")

    def augmentation_sequence(self):
        # Create an augmentation instance

        def fixFillMode(x, mode="reflect"):
            clsName = x.__class__.__name__
            # clsName = type(x).__name__
            if clsName == "CropAndPad":
                x._pad_mode_segmentation_maps = mode
            elif clsName == "Affine":
                x._mode_segmentation_maps = mode
            return x

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2),
                iaa.Sometimes(
                    0.5,
                    fixFillMode(
                        iaa.CropAndPad(
                            percent=(-0.05, 0.1), pad_mode="reflect", pad_cval=(0, 255)
                        )
                    ),
                ),
                iaa.Sometimes(
                    0.5,
                    fixFillMode(
                        iaa.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-25, 25),
                            shear=(-16, 16),
                            order=[0, 1],
                            cval=(0, 255),
                            mode="reflect",
                        )
                    ),
                ),
                iaa.SomeOf(
                    (0, 3),
                    [
                        iaa.OneOf(
                            [
                                iaa.GaussianBlur((0, 3.0)),
                                iaa.AverageBlur(k=(2, 7)),
                                iaa.MedianBlur(k=(3, 11)),
                            ]
                        ),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                        iaa.SimplexNoiseAlpha(
                            iaa.OneOf(
                                [
                                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                    iaa.DirectedEdgeDetect(
                                        alpha=(0.5, 1.0), direction=(0.0, 1.0)
                                    ),
                                ]
                            )
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        iaa.OneOf(
                            [
                                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                iaa.CoarseDropout(
                                    (0.03, 0.15),
                                    size_percent=(0.02, 0.05),
                                    per_channel=0.2,
                                ),
                            ]
                        ),
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.AddToHueAndSaturation((-20, 20)),
                        iaa.OneOf(
                            [
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-4, 0),
                                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                    second=iaa.ContrastNormalization((0.5, 2.0)),
                                ),
                            ]
                        ),
                        iaa.contrast.LinearContrast((0.75, 1.20), per_channel=0.5),
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        iaa.Sometimes(
                            0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                        ),
                        iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                    ],
                    random_order=True,
                ),
            ],
            random_order=True,
        )

        return seq


class MembraneDataGenerator(Iterator):
    # Generates data for Keras
    def __init__(
        self,
        source_dir,
        groundtruth_dir,
        target_shape,
        num_classes,
        batch_size,
        augmentation=False,
        shuffle=True,
        seed=None,
        **kwargs,
    ):

        # Initialization
        self.source_dir = source_dir
        self.groundtruth_dir = groundtruth_dir
        self.target_shape = tuple(target_shape)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augmentation = augmentation

        self.test = groundtruth_dir in ["", None]
        self.source_pairs = get_pairs_from_paths(
            source_dir, groundtruth_dir, test_time=self.test
        )

        super().__init__(len(self.source_pairs), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # Generates data containing batch_size samples
        images = [cv2.imread(self.source_pairs[i][0]) for i in index_array]

        if self.test:
            if self.augmentation:
                images = self.augmentation_sequence()(images=images)

            images_batch = self.process_images_batch(images)
            return images_batch

        # Load segmentations
        segmaps = [
            cv2.imread(self.source_pairs[i][1], flags=cv2.IMREAD_GRAYSCALE)[
                ..., np.newaxis
            ]
            for i in index_array
        ]

        # Do augmentations
        if self.augmentation:
            images, segmaps = self.augmentation_sequence()(
                images=images, segmentation_maps=segmaps
            )

        images_batch = self.process_images_batch(images)
        segmaps_batch = self.process_segmap_batch(segmaps)

        return images_batch, segmaps_batch

    def process_images_batch(self, images):
        return np.array(
            [rgb2gray(cv2.resize(img, self.target_shape[::-1])) / 255 for img in images]
        )[..., np.newaxis]

    def process_segmap_batch(self, segmaps):
        segmaps_batch = np.array(
            [
                cv2.resize(
                    seg, self.target_shape[::-1], interpolation=cv2.INTER_NEAREST
                )
                for seg in segmaps
            ]
        )
        background = segmaps_batch > 127.5
        segmaps_batch[background] = 0
        segmaps_batch[~background] = 1
        return segmaps_batch[..., np.newaxis]

    def augmentation_sequence(self):
        # Create an augmentation instance

        def fixFillMode(x, mode="reflect"):
            clsName = x.__class__.__name__
            # clsName = type(x).__name__
            if clsName == "CropAndPad":
                x._pad_mode_segmentation_maps = mode
            elif clsName == "Affine":
                x._mode_segmentation_maps = mode
            return x

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2),
                iaa.Sometimes(
                    0.5,
                    fixFillMode(
                        iaa.CropAndPad(
                            percent=(-0.05, 0.1), pad_mode="reflect", pad_cval=(0, 255)
                        )
                    ),
                ),
                iaa.Sometimes(
                    0.5,
                    fixFillMode(
                        iaa.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-45, 45),
                            shear=(-16, 16),
                            order=[0, 1],
                            cval=(0, 255),
                            mode="reflect",
                        )
                    ),
                ),
                iaa.SomeOf(
                    (0, 3),
                    [
                        iaa.OneOf(
                            [
                                iaa.GaussianBlur((0, 3.0)),
                                iaa.AverageBlur(k=(2, 7)),
                                iaa.MedianBlur(k=(3, 11)),
                            ]
                        ),
                        iaa.Sharpen(
                            alpha=(0, 1.0), lightness=(0.75, 1.5)
                        ),  # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.AddToHueAndSaturation((-20, 20)),
                        iaa.OneOf(
                            [
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-4, 0),
                                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                    second=iaa.ContrastNormalization((0.5, 2.0)),
                                ),
                            ]
                        ),
                        iaa.contrast.LinearContrast((0.75, 1.20), per_channel=0.5),
                        iaa.Sometimes(
                            0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                        ),
                        iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                    ],
                    random_order=True,
                ),
            ],
            random_order=True,
        )

        return seq


class DataGeneratorError(Exception):
    pass


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def get_pairs_from_paths(
    images_dir, segs_dir, ignore_non_matching=False, test_time=False
):
    """ Find all the images from the images_dir directory and
        the segmentation images from the segs_dir directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png"]

    pair_list = []

    for file in os.listdir(images_dir):
        image_full_path = os.path.join(images_dir, file)
        basename, ext = os.path.splitext(file)

        if os.path.isfile(image_full_path) and ext in ACCEPTABLE_IMAGE_FORMATS:
            if test_time:
                pair_list.append((image_full_path, ""))
                continue

            for ftype in ACCEPTABLE_SEGMENTATION_FORMATS:
                seg_full_path = os.path.join(segs_dir, basename + ftype)
                if os.path.isfile(seg_full_path):
                    pair_list.append((image_full_path, seg_full_path))
                    break
            else:
                if not ignore_non_matching:
                    raise DataGeneratorError(
                        "No corresponding segmentation found for image {0}.".format(
                            image_full_path
                        )
                    )

    return np.asarray(pair_list)


# def augmentation_sequence(self):
#         # Create an augmentation instance

#         def fixFillMode(x, mode="reflect"):
#             clsName = x.__class__.__name__
#             # clsName = type(x).__name__
#             if clsName == "CropAndPad":
#                 x._pad_mode_segmentation_maps = mode
#             elif clsName == "Affine":
#                 x._mode_segmentation_maps = mode
#             return x

#         seq = iaa.Sequential(
#             [
#                 # apply the following augmenters to most images
#                 iaa.Fliplr(0.5),  # horizontally flip 50% of all images
#                 iaa.Flipud(0.2),  # vertically flip 20% of all images
#                 # crop images by -5% to 10% of their height/width
#                 iaa.Sometimes(
#                     0.5,
#                     fixFillMode(
#                         iaa.CropAndPad(
#                             percent=(-0.05, 0.1), pad_mode="reflect", pad_cval=(0, 255)
#                         )
#                     ),
#                 ),
#                 iaa.Sometimes(
#                     0.5,
#                     fixFillMode(
#                         iaa.Affine(
#                             # scale images to 80-120% of their size, individually per axis
#                             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                             # translate by -20 to +20 percent (per axis)
#                             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#                             rotate=(-45, 45),  # rotate by -45 to +45 degrees
#                             shear=(-16, 16),  # shear by -16 to +16 degrees
#                             # use nearest neighbour or bilinear interpolation (fast)
#                             order=[0, 1],
#                             # if mode is constant, use a cval between 0 and 255
#                             cval=(0, 255),
#                             # use any of scikit-image's warping modes
#                             # (see 2nd image from the top for examples)
#                             mode="reflect",
#                         )
#                     ),
#                 ),
#                 # execute 0 to 3 of the following (less important) augmenters per
#                 # image don't execute all of them, as that would often be way too
#                 # strong
#                 iaa.SomeOf(
#                     (0, 4),
#                     [
#                         # convert images into their superpixel representation
#                         iaa.Sometimes(
#                             0.5,
#                             iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200)),
#                         ),
#                         iaa.OneOf(
#                             [
#                                 # blur images with a sigma between 0 and 3.0
#                                 iaa.GaussianBlur((0, 3.0)),
#                                 # blur image using local means with kernel sizes
#                                 # between 2 and 7
#                                 iaa.AverageBlur(k=(2, 7)),
#                                 # blur image using local medians with kernel sizes
#                                 # between 2 and 7
#                                 iaa.MedianBlur(k=(3, 11)),
#                             ]
#                         ),
#                         iaa.Sharpen(
#                             alpha=(0, 1.0), lightness=(0.75, 1.5)
#                         ),  # sharpen images
#                         iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
#                         # search either for all edges or for directed edges,
#                         # blend the result with the original image using a blobby mask
#                         iaa.SimplexNoiseAlpha(
#                             iaa.OneOf(
#                                 [
#                                     iaa.EdgeDetect(alpha=(0.5, 1.0)),
#                                     iaa.DirectedEdgeDetect(
#                                         alpha=(0.5, 1.0), direction=(0.0, 1.0)
#                                     ),
#                                 ]
#                             )
#                         ),
#                         # add gaussian noise to images
#                         iaa.AdditiveGaussianNoise(
#                             loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
#                         ),
#                         iaa.OneOf(
#                             [
#                                 # randomly remove up to 10% of the pixels
#                                 iaa.Dropout((0.01, 0.1), per_channel=0.5),
#                                 iaa.CoarseDropout(
#                                     (0.03, 0.15),
#                                     size_percent=(0.02, 0.05),
#                                     per_channel=0.2,
#                                 ),
#                             ]
#                         ),
#                         # invert color channels
#                         # iaa.Invert(0.05, per_channel=True),
#                         # change brightness of images (by -10 to 10 of original value)
#                         iaa.Add((-10, 10), per_channel=0.5),
#                         # change hue and saturation
#                         iaa.AddToHueAndSaturation((-20, 20)),
#                         # either change the brightness of the whole image (sometimes
#                         # per channel) or change the brightness of subareas
#                         iaa.OneOf(
#                             [
#                                 iaa.Multiply((0.5, 1.5), per_channel=0.5),
#                                 iaa.FrequencyNoiseAlpha(
#                                     exponent=(-4, 0),
#                                     first=iaa.Multiply((0.5, 1.5), per_channel=True),
#                                     second=iaa.ContrastNormalization((0.5, 2.0)),
#                                 ),
#                             ]
#                         ),
#                         # improve or worsen the contrast
#                         iaa.contrast.LinearContrast((0.75, 1.20), per_channel=0.5),
#                         iaa.Grayscale(alpha=(0.0, 1.0)),
#                         # move pixels locally around (with random strengths)
#                         iaa.Sometimes(
#                             0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
#                         ),
#                         # sometimes move parts of the image around
#                         iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
#                         iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
#                     ],
#                     random_order=True,
#                 ),
#             ],
#             random_order=True,
#         )

#         return seq
