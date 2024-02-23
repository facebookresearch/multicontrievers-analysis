from typing import Union, List


import numpy as np
from datasets import ClassLabel
import torch
from torch.utils.data import TensorDataset
from abc import ABC, abstractmethod
from scipy.sparse._csr import csr_matrix

import ipdb


class Data(ABC):
    """
    An abstract class for loading and arranging data from files.
    """

    def __init__(self):
        self.dataset = None


class BasicDataset(Data):
    def __init__(self, train, dev, test, label_key='z', no_dev: bool = False, 
                dataset_name="biasinbios"):
        self.train = torch.load(train, map_location='cuda')
        self.test = torch.load(test, map_location='cuda')
        
        if no_dev:  # combine dev and test
            try:
                dev = torch.load(dev, map_location='cuda')
                self.test["X"] = torch.cat([self.test["X"], dev["X"]])
                self.test["y"] = np.concatenate([self.test["y"], dev["y"]])
                self.test["z"] = np.concatenate([self.test["z"], dev["z"]])
            except:
                print("No dev to load in")
            self.dev = {}
        else:
            self.dev = torch.load(dev, map_location='cuda')
        self.label_key = label_key
        self.dataset_name = dataset_name
        self.set_n_labels()
        if dataset_name == "biasinbios":
            if label_key == "z":
                self.labels = ClassLabel(names=["m", "f"])  # to ensure always m:0 f:1
            elif label_key == "y":
                self.labels = ClassLabel(names=self.unique_labels)
    
    def set_n_labels(self):
        unique_labels = set(self.train.get(self.label_key)) | set(self.dev.get(self.label_key, [])) | set(self.train.get(self.label_key))
        self.n_labels = len(unique_labels)
        self.unique_labels = unique_labels    
    
    def _preprocess_probing_data(self, split):
        if len(split) == 0:
            return []
        X, z = split['X'], split[self.label_key]
        if self.label_key == 'y':  # use y labels to probe instead of the protected attribute z
            z = split["y"]
            
        if self.dataset_name == "biasinbios":
            for l in self.unique_labels:
                z[z == l] = self.labels.str2int(l)
        z = z.astype(int)
        z = torch.tensor(z).short()
        if type(X) != torch.Tensor:
            if type(X) != csr_matrix:
                X = torch.tensor(X)
            else:
                row_idx, col_idx = X.nonzero()
                X = torch.sparse_csr_tensor(X.indptr, col_idx, X.data)
        assert X.dim() == 2, f"shape of input vectors is not 2D, this is a problem. Shape: {X.shape}"

        return TensorDataset(X, z)

    def preprocess_probing_data(self):
        self.train = self._preprocess_probing_data(self.train)
        self.dev = self._preprocess_probing_data(self.dev)
        self.test = self._preprocess_probing_data(self.test)


class BiasInBiosData(Data):
    def __init__(self, path: Union[str, List[str]], seed, split, balanced):
        super().__init__()
        self.dataset = None
        self.original_y = None
        self.original_z = None
        self.n_labels = 0
        self.z = None
        self.load_bias_in_bios_dataset(path, seed, split, balanced)
        self.perc = self.compute_perc()

    @abstractmethod
    def load_bias_in_bios_dataset(self, path: str, seed, split, balanced=None):
        ...

    def get_label_to_code(self):
        code_to_label = dict(enumerate(self.cat.categories))
        label_to_code = {v: k for k, v in code_to_label.items()}
        return label_to_code

    def compute_perc(self):
        perc = {}
        golden_y = self.original_y
        for profession in np.unique(golden_y):
            total_of_label = len(golden_y[golden_y == profession])
            indices_female = np.logical_and(golden_y == profession, self.original_z == 'F')
            perc_female = len(golden_y[indices_female]) / total_of_label
            perc[profession] = perc_female

        return perc
        # professions_bios_stats = {'accountant': 59.7,
        #                           'architect': 16.5,
        #                           'attorney': 37.4,
        #                           'chiropractor': 22.2,
        #                           'composer': 30.8,
        #                           'dentist': 28.7,
        #                           'dietitian': 91.4,
        #                           'filmmaker': 45.2,
        #                           'interior_designer': 0.8027397260273973,
        #                           'journalist': 54.0,
        #                           'nurse': 87.4,
        #                           'painter': 53.5,  # artists
        #                           'paralegal': 85.8,
        #                           'pastor': 71.5,  # religious workers, all others (in service occupation)
        #                           'personal_trainer': 67.1,
        #                           'photographer': 52.1,
        #                           'physician': 40.6,
        #                           'poet': 30.8,
        #                           'psychologist': 80.3,
        #                           'software_engineer': 19.4,
        #                           'surgeon': 40.6,
        #                           'teacher': 73.5,  # Education, training, and library occupations
        #                           }
        # return professions_bios_stats

# class BiasInBiosDataFinetuning(BiasInBiosData):

#     def split_and_balance(self, path, seed, split, balanced=None, other=None):
#         data = BiasinBios_split_and_return_tokens_data(seed, path, other=other)
#         self.cat = data["categories"]

#         X, y, masks, other = data[split]["X"], data[split]["y"], data[split]["masks"], data[split]["other"]

#         self.z = data[split]["z"]
#         self.original_z = data[split]["z"]

#         if balanced in ("oversampled", "subsampled"):
#             output = balance_dataset(X, y, self.z, masks=masks, other=other,
#                                                   oversampling=True if balanced == "oversampled" else False)
#             X, y, self.z, masks = output[0], output[1], output[2], output[3]
#             if other is not None:
#                 other = output[4]

#         self.cat = data["categories"]
#         y = torch.tensor(y).long()
#         X = torch.tensor(X).long()
#         masks = torch.tensor(masks).long()

#         self.original_y = data[split]["original_y"]
#         self.n_labels = len(np.unique(self.original_y))

#         # torch.save({"X": X, "y": y, "masks": masks, "z": self.z, "original_y": self.original_y, "original_z": self.original_z, "n_labels": self.n_labels, "cat": self.cat}
#         # ,f"../data/biasbios/bias_in_bios_{split}_raw_oversampled.pt")
#         # print("SAVED $$$$$$$$$$$$$$$$$$$$$")

#         if other is not None:
#             return X, y, masks, other
#         else:
#             return X, y, masks

#         # data = torch.load(path)
#         #
#         # self.original_y = data["original_y"]
#         # self.original_z = data["original_z"]
#         # self.n_labels = data["n_labels"]
#         # self.cat = data["cat"]
#         # self.z = data["z"]
#         #
#         #
#         # if "other" in data:
#         #     return data["X"], data["y"], data["masks"], data["other"]
#         # else:
#         #     return data["X"], data["y"], data["masks"]

#     def load_bias_in_bios_dataset(self, path: str, seed, split, balanced=None):

#         X, y, masks = self.split_and_balance(path, seed, split, balanced)

#         self.dataset = TensorDataset(X, y, masks)

# class BiasInBiosDataPoE(BiasInBiosDataFinetuning):

#     def load_bias_in_bios_dataset(self, path: List[str], seed, split, balanced=None):

#         assert(len(path) == 2)

#         X, y, masks = self.split_and_balance(path[0], seed, split, balanced)
#         X_b, _, masks_b = self.split_and_balance(path[1], seed, split, balanced)

#         self.dataset = TensorDataset(X, y, masks, X_b, masks_b)

# class BiasInBiosDataPoEScrubbedInfo(BiasInBiosData):

#     def split_and_balance(self, path, biased_features_path, seed, split, balanced=None):
#         biased_features = torch.load(biased_features_path)
#         data = BiasinBios_split_and_return_tokens_data(seed, path, other=biased_features)
#         self.cat = data["categories"]
#         X, y, masks, X_b = data[split]["X"], data[split]["y"], data[split]["masks"], data[split]["other"]

#         self.z = data[split]["z"]
#         self.original_z = data[split]["z"]

#         if balanced in ("oversampled", "subsampled"):
#             X, y, self.z, masks = balance_dataset(X, y, self.z, masks=masks, other=[X_b],
#                                                   oversampling=True if balanced == "oversampled" else False)

#         self.cat = data["categories"]
#         y = torch.tensor(y).long()
#         X = torch.tensor(X).long()
#         masks = torch.tensor(masks).long()
#         X_b = torch.tensor(X_b).float()

#         self.original_y = data[split]["y"]
#         self.z = data[split]["z"]
#         self.n_labels = len(np.unique(self.original_y))

#         return X, y, masks, X_b

#     def load_bias_in_bios_dataset(self, path: List[str], seed, split, balanced=None):
#         assert (len(path) == 2)
#         X, y, masks, X_b = self.split_and_balance(path[0], path[1], seed, split, balanced)

#         self.dataset = TensorDataset(X, y, masks, X_b)

# class BiasInBiosDataDFL(BiasInBiosDataFinetuning):

#     def load_bias_in_bios_dataset(self, path: List[str], seed, split, balanced=None):

#         assert(isinstance(path, str))

#         X, y, masks = self.split_and_balance(path, seed, split, balanced)
#         z = self.z
#         z = pd.Categorical(z).codes
#         z = torch.tensor(z).long().view(-1, 1)

#         self.dataset = TensorDataset(X, y, masks, z)

# class BiasInBiosDataDFLHard(BiasInBiosDataFinetuning):

#     def load_bias_in_bios_dataset(self, path: List[str], seed, split, balanced=None):

#         assert(isinstance(path, list))
#         assert(len(path) == 2)

#         z_preds = torch.tensor(torch.load(path[1]))
#         X, y, masks, z_preds = self.split_and_balance(path[0], seed, split, balanced, z_preds)
#         z = self.z
#         z = pd.Categorical(z).codes
#         z = torch.tensor(z).long().view(-1, 1)

#         self.dataset = TensorDataset(X, y, masks, z, z_preds)

# class BiasInBiosDataLinear(BiasInBiosData):

#     def load_bias_in_bios_dataset(self, path: str, seed, split, balanced=None):
#         data = BiasinBios_split_and_return_vectors_data(seed, path)
#         self.cat = data["categories"]
#         X, y = data[split]["X"], data[split]["y"]

#         self.z = data[split]["z"]
#         self.original_z = data[split]["z"]

#         if balanced in ("oversampled", "subsampled"):
#             X, y, self.z = balance_dataset(X, y, self.z, oversampling=True if balanced == "oversampled" else False)

#         y = torch.tensor(y).long()
#         X = torch.tensor(X)

#         self.dataset = TensorDataset(X, y)
#         self.original_y = data[split]["original_y"]
#         self.n_labels = len(np.unique(self.original_y))

#     # def get_label_to_code(self):
#     #     code_to_label = dict(enumerate(self.cat.categories))
#     #     label_to_code = {v: k for k, v in code_to_label.items()}
#     #     return label_to_code


# class BiasInBiosDataLM(Data):
#     def __init__(self, path: string, load_test: bool):
#         super().__init__()
#         # true mask length
#         self.dataset = None
#         # different masks lengths
#         self.test_dataset = None
#         # self.dataset_3 = None
#         # self.dataset_4 = None
#         # self.true_length = None
#         self.numerical_labels = None
#         self.z = None
#         self.original_y = None
#         self.load_test = load_test
#         self.load_bias_in_bios_dataset(path)

#     def load_bias_in_bios_dataset(self, path: str):
#         def extract_data(data_dict):
#             X, y, masks = data_dict["X"], data_dict["y"], data_dict["masks"]
#             # print(y)
#             # y = torch.Tensor(y).to(device).long()
#             y = torch.Tensor(y).long()
#             # X = torch.Tensor(X).to(device).long()
#             X = torch.Tensor(X).long()
#             # masks = torch.Tensor(masks).to(device).long()
#             masks = torch.Tensor(masks).long()
#             return X, y, masks

#         # print(path)
#         data = torch.load(path)

#         if self.load_test:
#             pos = str.find(path, "-seed")
#             data_2 = torch.load(path[:pos] + "_mask_length_2" + path[pos:])
#             data_3 = torch.load(path[:pos] + "_mask_length_3" + path[pos:])
#             data_4 = torch.load(path[:pos] + "_mask_length_4" + path[pos:])

#         true_length = torch.Tensor(data['true_length']).long()
#         self.z = data['z']
#         self.original_y = data['labels']
#         labels = pd.Categorical(data['labels']).codes
#         labels = torch.Tensor(labels).long()
#         self.numerical_labels = labels

#         self.dataset = TensorDataset(*extract_data(data))

#         if self.load_test:
#             self.test_dataset = TensorDataset(*extract_data(data_2), *extract_data(data_3), *extract_data(data_4), true_length)

#     def get_label_to_code(self):
#         cat = pd.Categorical(self.original_y)
#         code_to_label = dict(enumerate(cat.categories))
#         label_to_code = {v: k for k, v in code_to_label.items()}
#         return label_to_code
