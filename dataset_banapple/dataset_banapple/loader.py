import argparse
import enum
import inspect
import os
import os.path
from dataclasses import dataclass
from typing import Optional

import cv2
import pandas

# Name of the metadata file that should come with the banapple dataset.
METADATA_FILE = 'metadata.csv'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=os.path.abspath,
                        help=f"Path to the dataset. This folder should contain {METADATA_FILE}.")
    parser.add_argument("sample_size", type=int, default=-1,
                        help="Max. number of samples to load.")
    return parser.parse_args()


class BanappleClass(enum.IntFlag):
    """

    Categories that a banapple datapoint could fit into.

    Currently, Nothing and Both are unused.
    """
    Nothing = 0
    Apple = enum.auto()
    Banana = enum.auto()
    Both = Apple | Banana


@dataclass
class DataPoint:
    """
    Describes a single example in the dataset.
    """
    name: str
    full_path: os.PathLike
    cls: BanappleClass
    num_apples: int
    num_bananas: int
    owner: str
    
    # Do not access _image directly; use self.image instead!
    _image: Optional[cv2.typing.MatLike] = None

    @property
    def image(self) -> cv2.typing.MatLike:
        """
        Loads image data only when requested.
        """
        if not self._image:
            self._image = cv2.imread(self.full_path)
        return self._image

    @property
    def vector(self) -> cv2.typing.MatLike:
        """
        Transforms image to a 1D vector.
        """
        return self.image.reshape((self.image.size, 1))

    def __hash__(self) -> int:
        return hash((self.name, self.full_path, self.cls,
                     self.num_apples, self.num_bananas, self.owner))

    def __setattr__(self, key: str, value):
        """
        Treat all "public" values as frozen.

        A frozen field cannot be changed after this object is created.  That
        is just like with data inside a tuple: that data should not change 
        as well.
        
        This class treats all fields that do not start with "_" as public.
        Fields starting with "_" should not be used by objects other than self.
        """
        stack = inspect.stack()
        if len(stack) < 2 or stack[1].function != '__init__':
            if not key.startswith("_"):
                raise RuntimeError(f"Forbidden to change value of member {key}.")
        super().__setattr__(key, value)


def load_dataset(path: os.PathLike, sample_size=-1) -> set[DataPoint]:
    """
    Load dataset and its metadata.

    If sample_size is a non-negative integer, return at most that number of DataPoints.
    Otherwise, return all DataPoints.
    """
    if not os.path.isdir(path):
        raise RuntimeError(f"Path '{path}' does not point to a folder.")
    if not os.path.isfile(os.path.join(path, METADATA_FILE)):
        raise RuntimeError(f"Expected folder '{path}' to contain file {METADATA_FILE}.")

    dataset: set[DataPoint] = set()

    metadata = pandas.read_csv(os.path.join(path, METADATA_FILE))
    if 0 <= sample_size <= len(metadata):
        metadata = metadata.sample(sample_size)
    for _, row in metadata.iterrows():
        contains_apple = row['contains_apple'] > 0
        contains_banana = row['contains_banana'] > 0
        if contains_apple == contains_banana:
            # This datapoint is hard to work with when doing 
            # binary classification.  For now, we skip it.
            continue
        elif contains_apple:
            cls = BanappleClass.Apple
        elif contains_banana:
            cls = BanappleClass.Banana
        full_path = os.path.join(path,
                                 os.path.extsep.join([row['filename'], 'jpg']))

        datum = DataPoint(name=row['filename'],
                          full_path=full_path,
                          cls=cls,
                          num_apples=row['contains_apple'],
                          num_bananas=row['contains_banana'],
                          owner=row['owner'])
        dataset.add(datum)
    return dataset


if __name__ == "__main__":
    args = parse_arguments()
    dataset = load_dataset(args.dataset_dir, args.sample_size)
    print(f"Collected {len(dataset)} data points:")
    for datum in dataset:
        print(f"  - {datum.name}")