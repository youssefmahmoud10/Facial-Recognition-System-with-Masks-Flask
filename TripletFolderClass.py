T_G_SEED = 1337

#Torch Imports
import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random
import numpy as np

np.random.seed(T_G_SEED)
torch.manual_seed(T_G_SEED)
random.seed(T_G_SEED)

# Define the DataSet object to load the data from folders.
# Inherit from the PyTorch ImageFolder class, which gets us close to
# what we need. Necessary changes are to create an inverse look-up table
# based on labels. Given a label, find another random image with that
# same label, and also take a random image from a random other different 
# category for a negative instance, giving us the triplet: 
# [anchor, positive, negative].
class TripletFolder(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super(TripletFolder, self).__init__(root=root, transform=transform)

        # Create a dictionary of lists for each class for reverse lookup
        # to generate triplets 
        self.classdict = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdict[ci] = []

        # append each file in the approach dictionary element list
        for s in self.samples:
            self.classdict[s[1]].append(s[0])

        # keep track of the sizes for random sampling
        self.classdictsize = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdictsize[ci] = len(self.classdict[ci])

    # Return a triplet, with positive and negative selected at random.
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, sample, sample) where the samples are anchor, positive, and negative.
            The positive and negative instances are sampled randomly. 
        """

        # The anchor is the image at this index.
        #a_path = image_path , a_target = label()
        a_path, a_target = self.samples[index]

        prs = random.random() # positive random sample
        nrc = random.random() # negative random class
        nrs = random.random() # negative random sample

        # random negative class cannot be the same class as anchor. We add
        # a random offset that is 1 less than the number required to wrap
        # back around to a_target after modulus. 
        nrc = (a_target + int(nrc*(len(self.classes) - 1))) % len(self.classes)

        # Positive Instance: select a random instance from the same class as anchor.
        # el path beta3 el positive image hyb2a el label beta3 el class fi kam soora 
        # wel 3uddud dugh yederreb fih (0-1) 3shan mutetlu3sh brra el range.  
        p_path = self.classdict[a_target][int(self.classdictsize[a_target]*prs)]
        
        # Negative Instance: select a random instance from the random negative class.
        n_path = self.classdict[nrc][int(self.classdictsize[nrc]*nrs)]

        # Load the data for these samples.
        a_sample = self.loader(a_path)
        p_sample = self.loader(p_path)
        n_sample = self.loader(n_path)

        # apply transforms
        if self.transform is not None:
            a_sample = self.transform(a_sample)
            p_sample = self.transform(p_sample)
            n_sample = self.transform(n_sample)

        # note that we do not return the label! 
        return a_sample, p_sample, n_sample, a_target