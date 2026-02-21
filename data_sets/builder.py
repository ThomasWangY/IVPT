"""
Dataset builder for IVPT.

Provides the ``get_dataset`` function to construct training and test
datasets based on command-line arguments.
"""

from data_sets.fg_bird_dataset import FineGrainedBirdClassificationDataset


def get_dataset(args, train_transforms, test_transforms):
    if args.dataset == 'cub' or args.dataset == 'nabirds':
        dataset_train = FineGrainedBirdClassificationDataset(args.data_path, split=args.train_split, mode='train',
                                                             transform=train_transforms,
                                                             image_sub_path=args.image_sub_path_train)
        dataset_test = FineGrainedBirdClassificationDataset(args.data_path, mode=args.eval_mode,
                                                            transform=test_transforms,
                                                            image_sub_path=args.image_sub_path_test)
        num_cls = dataset_train.num_classes
    else:
        raise ValueError('Dataset not supported.')
    return dataset_train, dataset_test, num_cls
