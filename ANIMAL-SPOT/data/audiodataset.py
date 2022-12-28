"""
Module: audiodataset.py
Authors: Christian Bergler, Hendrik Schroeter
GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import os
import sys
import csv
import glob
import random
import pathlib
import platform
import numpy as np
import soundfile as sf
import data.transforms as T

import torch
import torch.utils.data
import torch.multiprocessing as mp

from math import ceil
from types import GeneratorType
from utils.logging import Logger
from collections import defaultdict
from utils.FileIO import AsyncFileReader
from typing import Any, Dict, Iterable, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Data preprocessing default options
"""
DefaultSpecDatasetOps = {
    "sr": 44100,
    "preemphases": 0.98,
    "n_fft": 4096,
    "hop_length": 441,
    "n_freq_bins": 256,
    "fmin": 500,
    "fmax": 10000,
    "freq_compression": "linear",
    "min_level_db": -100,
    "ref_level_db": 20,
}

"""
Get audio files from directory
"""
def get_audio_files_from_dir(path: str):
    audio_files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    audio_files = map(lambda p: pathlib.Path(p), audio_files)
    audio_files = filter(lambda p: not p.match("*.bkp/*"), audio_files)
    base = pathlib.Path(path)
    return map(lambda p: str(p.relative_to(base)), audio_files)

"""
Helper class in order to speed up filtering potential broken files
"""
class _FilterPickleHelper(object):
    def __init__(self, predicate, *pred_args):
        self.predicate = predicate
        self.args = pred_args

    def __call__(self, item):
        return self.predicate(item, *self.args)

"""
Parallel Filtering to analyze incoming data files
"""
class _ParallelFilter(object):
    def __init__(self, iteratable, n_threads=None, chunk_size=1):
        self.data = iteratable
        self.n_threads = n_threads
        self.chunk_size = chunk_size

    def __call__(self, func, *func_args):
        with mp.Pool(self.n_threads) as pool:
            func_pickle = _FilterPickleHelper(func, *func_args)
            for keep, c in pool.imap_unordered(func_pickle, self.data, self.chunk_size):
                if keep:
                    yield c

"""
Analyzing loudness criteria of each audio file by checking maximum amplitude (default: 1e-3)
"""
def _loudness_criteria(file_name: str, working_dir: str = None):
    if working_dir is not None:
        file_path = os.path.join(working_dir, file_name)
    else:
        file_path = file_name
    y, __ = sf.read(file_path, always_2d=True, dtype="float32")
    max_ampl = y.max()
    if max_ampl < 1e-3:
        return True, file_name
    else:
        return False, None

"""
Filtering all audio files in previous which do not fulfill the loudness criteria
"""
def get_broken_audio_files(files: Iterable[str], working_dir: str = None):
    f = _ParallelFilter(files, chunk_size=100)
    return f(_loudness_criteria, working_dir)


"""
Computes the CSV Split in order to prepare for randomly partitioning all data files into a training, validation, and test corpus
by dividing the data in such a way that audio files of a given tape are stored only in one of the three partitions.
The filenames per dataset will be stored in CSV files (train.csv, val.csv, test.csv). Each CSV File will be merged into
a train, val, and test file holding the information how a single partition is made up from single CSV files. These three
files reflects the training, validation, and test set.
"""
class CsvSplit(object):

    def __init__(
        self,
        split_fracs: Dict[str, float],
        working_dir: (str) = None,
        seed: (int) = None,
        split_per_dir=False,
    ):
        if not np.isclose(np.sum([p for _, p in split_fracs.items()]), 1.):
            raise ValueError("Split probabilities have to sum up to 1.")
        self.split_fracs = split_fracs
        self.working_dir = working_dir
        self.seed = seed
        self.split_per_dir = split_per_dir
        self.splits = defaultdict(list)
        self._logger = Logger("CSVSPLIT")

    """
    Return split for given partition. If there is already an existing CSV split return this split if it is valid or
    in case there exist not a split yet generate a new CSV split
    """
    def load(self, split: str, files: List[Any] = None):

        if split not in self.split_fracs:
            raise ValueError(
                "Provided split '{}' is not in `self.split_fracs`.".format(split)
            )

        if self.splits[split]:
            return self.splits[split]
        if self.working_dir is None:
            self.splits = self._split_with_seed(files)
            return self.splits[split]
        if self.can_load_from_csv():
            if not self.split_per_dir:
                csv_split_files = {
                    split_: (os.path.join(self.working_dir, split_ + ".csv"),)
                    for split_ in self.split_fracs.keys()
                }
            else:
                csv_split_files = {}
                for split_ in self.split_fracs.keys():
                    split_file = os.path.join(self.working_dir, split_)
                    csv_split_files[split_] = []
                    with open(split_file, "r") as f:
                        for line in f.readlines():
                            csv_split_files[split_].append(line.strip())

            for split_ in self.split_fracs.keys():
                for csv_file in csv_split_files[split_]:
                    if not csv_file or csv_file.startswith(r"#"):
                        continue
                    csv_file_path = os.path.join(self.working_dir, csv_file)
                    with open(csv_file_path, "r") as f:
                        reader = csv.reader(f)
                        for item in reader:
                            file_ = os.path.basename(item[0])
                            file_ = os.path.join(os.path.dirname(csv_file), file_)
                            self.splits[split_].append(file_)
            return self.splits[split]

        if not self.split_per_dir:
            working_dirs = (self.working_dir,)
        else:
            f_d_map = self.get_f_d_map(files)
            working_dirs = [os.path.join(self.working_dir, p) for p in f_d_map.keys()]
        for working_dir in working_dirs:
            splits = self._split_with_seed(
                files if not self.split_per_dir else f_d_map[working_dir]
            )
            for split_ in splits.keys():
                csv_file = os.path.join(working_dir, split_ + ".csv")
                self._logger.debug("Generating {}".format(csv_file))
                if self.split_per_dir:
                    with open(os.path.join(self.working_dir, split_), "a") as f:
                        p = pathlib.Path(csv_file).relative_to(self.working_dir)
                        f.write(str(p) + "\n")
                if len(splits[split_]) == 0:
                    raise ValueError(
                        "Error splitting dataset. Split '{}' has 0 entries".format(
                            split_
                        )
                    )
                with open(csv_file, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    for item in splits[split_]:
                        writer.writerow([item])
                self.splits[split_].extend(splits[split_])
        return self.splits[split]

    """
    Check whether it is possible to correctly load information from existing csv files
    """
    def can_load_from_csv(self):
        if not self.working_dir:
            return False
        if self.split_per_dir:
            for split in self.split_fracs.keys():
                split_file = os.path.join(self.working_dir, split)
                if not os.path.isfile(split_file):
                    return False
                self._logger.debug("Found dataset split file {}".format(split_file))
                with open(split_file, "r") as f:
                    for line in f.readlines():
                        csv_file = line.strip()
                        if not csv_file or csv_file.startswith(r"#"):
                            continue
                        if not os.path.isfile(os.path.join(self.working_dir, csv_file)):
                            self._logger.error("File not found: {}".format(csv_file))
                            raise ValueError(
                                "Split file found, but csv files are missing. "
                                "Aborting..."
                            )
        else:
            for split in self.split_fracs.keys():
                csv_file = os.path.join(self.working_dir, split + ".csv")
                if not os.path.isfile(csv_file):
                    return False
                self._logger.debug("Found csv file {}".format(csv_file))
        return True

    """
    Create a mapping from directory to containing files.
    """
    def get_f_d_map(self, files: List[Any]):

        f_d_map = defaultdict(list)
        if self.working_dir is not None:
            for f in files:
                f_d_map[str(pathlib.Path(self.working_dir).joinpath(f).parent)].append(
                    f
                )
        else:
            for f in files:
                f_d_map[str(pathlib.Path(".").resolve().joinpath(f).parent)].append(f)
        return f_d_map

    """
    Randomly splits the dataset using given seed
    """
    def _split_with_seed(self, files: List[Any]):
        if not files:
            raise ValueError("Provided list `files` is `None`.")
        if self.seed:
            random.seed(self.seed)
        return self.split_fn(files)

    """
    A generator function that returns all values for the given `split`.
    """
    def split_fn(self, files: List[Any]):
        _splits = np.split(
            ary=random.sample(files, len(files)),
            indices_or_sections=[
                int(p * len(files)) for _, p in self.split_fracs.items()
            ],
        )
        splits = dict()
        for i, key in enumerate(self.splits.keys()):
            splits[key] = _splits[i]
        return splits

"""
Extracts the year and tape from the given audio filename (filename structure: call-label_ID_YEAR_TAPE_STARTTIME_ENDTIME)
"""
def get_tape_key(file, valid_years=None):
    while "__" in file:
        file = file.replace("__", "_")
    try:
        attributes = file.split(sep="_")
        year = attributes[-4]
        tape = attributes[-3]
        if valid_years is not None and int(year) not in valid_years:
            return None
        return year + "_" + tape.upper()
    except Exception:
        import traceback
        print("Warning: skippfing file {}\n{}".format(file, traceback.format_exc()))
        pass
    return None


"""
Splits a given list of file names across different partitions.
"""
class DatabaseCsvSplit(CsvSplit):
    valid_years = set(range(1950, 2200))

    """
    Count the samples per tape.
    """
    def split_fn(self, files: Iterable[Any]):
        if isinstance(files, GeneratorType):
            files = list(files)
        n_files = len(files)
        tapes = defaultdict(int)
        for file in files:
            try:
                key = get_tape_key(file, self.valid_years)
                if key is not None:
                    tapes[key] += 1
                else:
                    n_files -= 1
            except IndexError:
                n_files -= 1
                pass

        tape_names = list(tapes)

        """
        Helper class which creates a mapping (per fraction) in order to handle added tapes and number of files per tape
        """
        class Mapping:
            def __init__(self):
                self.count = 0
                self.names = []

            def add(self, name, count):
                self.count += count
                self.names.append(name)

        mappings = {s: Mapping() for s in self.split_fracs.keys()}

        for tape_name in tape_names:
            missing_files = {
                s: n_files * f - mappings[s].count for s, f in self.split_fracs.items()
            }
            r = random.uniform(0., sum(f for f in missing_files.values()))
            for _split, _n_files in missing_files.items():
                r -= _n_files
                if r < 0:
                    mappings[_split].add(tape_name, tapes[tape_name])
                    break
            assert r < 0, "Should not get here"

        splits = defaultdict(list)
        for file in files:
            tape = get_tape_key(file, self.valid_years)
            if tape is not None:
                for s, m in mappings.items():
                    if tape in m.names:
                        splits[s].append(file)

        return splits

"""
Dataset for that returns just the provided file names.
"""
class FileNameDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir=None,
        transform=None,
        logger_name="TRAIN",
        dataset_name=None,
    ):
        if isinstance(file_names, GeneratorType):
            self.file_names = list(file_names)
        else:
            self.file_names = file_names
        self.working_dir = working_dir
        self.transform = transform
        self._logger = Logger(logger_name)
        self.dataset_name = dataset_name

    def __len__(self):
        if not isinstance(self.file_names, list):
            self.file_names = list(self.file_names)
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.working_dir:
            return os.path.join(self.working_dir, self.file_names[idx])
        sample = self.file_names[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

"""
Dataset for loading audio data.
"""
class AudioDataset(FileNameDataset):

    def __init__(
        self,
        file_names: Iterable[str],
        working_dir=None,
        sr=44100,
        mono=True,
        *args,
        **kwargs
    ):
        super().__init__(file_names, working_dir, *args, **kwargs)

        self.sr = sr
        self.mono = mono

    def __getitem__(self, idx):
        file = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file)
        sample = T.load_audio_file(file, self.sr, self.mono)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

"""
Dataset to load audio files of each partition and compute several data preprocessing steps (resampling, spectrogram transformation, augmentation, compression, normalization, subsampling/padding, etc.).
Filenames have to fulfill the follwing structure in order to ensure a correct data processing:  

Binary Segmentation: target/noise-*ADDITIONAL~LABEL~INFO*_ID_YEAR_TAPENAME_STARTTIME_ENDTIME
Multi-Class Classification: *LABEL*-*ADDITIONAL~LABEL~INFO*_ID_YEAR_TAPENAME_STARTTIME_ENDTIME

Single components of the filename template:
--------------------------------------------
1. target/noise-*ADDITIONAL~LABEL~INFO* = target-*ADDITIONAL~LABEL~INFO* or noise-*ADDITIONAL~LABEL~INFO* for target and any kind of noise signals. *ADDITIONAL~LABEL~INFO* 
is a placeholder for any kind of additional label information, e.g. target-alarmcall, noise-electricboatnoise
2. ID = unique ID to identify the audio clip
3. YEAR = year when the tape was recorded
4. TAPENAME = name of the recorded tape (has to be unique in order to do a proper data split into train, devel, test set by putting one tape only in one of the three sets)
5. STARTTIME = start time of the clip in milliseconds (integer number, e.g 123456ms = 123.456s)
5. ENDTIME = end time of the clip in milliseconds

The underscore "_" works as separator between the components and should not be included except for separation purposes. 
"""
class Dataset(AudioDataset):

    """
    Create variables in order to filter the filenames whether it is a target signal (call) or a noise signal (noise). Moreover
    the entire spectral transform pipeline is created in oder to set up the data preprocessing for each audio file.
    """
    def __init__(
        self,
        file_names: Iterable[str],
        working_dir=None,
        cache_dir=None,
        classes=None,
        sr=44100,
        n_fft=4096,
        hop_length=441,
        freq_compression="linear",
        n_freq_bins=256,
        f_min=0,
        f_max=18000,
        seq_len=128,
        augmentation=False,
        noise_files=[],
        min_max_normalize=False,
        *args,
        **kwargs
    ):
        super().__init__(file_names, working_dir, sr, *args, **kwargs)
        if self.dataset_name is not None:
            self._logger.info("Init dataset {}...".format(self.dataset_name))

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.classes = classes
        self.num_classes = len(classes)
        self.n_freq_bins = n_freq_bins

        valid_freq_compressions = ["linear", "mel", "mfcc"]

        if freq_compression not in valid_freq_compressions:
            raise ValueError(
                "{} is not a valid freq_compression. Must be one of {}",
                format(freq_compression, valid_freq_compressions),
            )
        self.freq_compression = freq_compression


        self._logger.debug(
            "Number of files : {}".format(len(self.file_names))
        )

        self.class_dist_dict = dict()

        if self.num_classes == 2:
            for class_val in self.classes:
                if class_val == "target":
                    self.class_dist_dict[class_val] = 1
                else:
                    self.class_dist_dict[class_val] = 0
        else:
            for class_idx in range(len(self.classes)):
                self.class_dist_dict[self.classes[class_idx]] = class_idx

        calls = defaultdict(int)

        for f in self.file_names:
            calls[self.get_class_type(f)] += 1

        for c_t, n in calls.items():
            self._logger.debug(
                "Number of samples in " + self.dataset_name + " for {}: {}".format(self.get_class_type_from_idx(c_t), n)
            )

        self.augmentation = augmentation

        spec_transforms = [
            lambda fn: T.load_audio_file(fn, sr=sr),
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.Spectrogram(n_fft, hop_length, center=False),
        ]

        self.file_reader = AsyncFileReader()

        if cache_dir is None:
            self.t_spectrogram = T.Compose(spec_transforms)
        else:
            self.t_spectrogram = T.CachedSpectrogram(
                cache_dir=cache_dir,
                spec_transform=T.Compose(spec_transforms),
                n_fft=n_fft,
                hop_length=hop_length,
                file_reader=self.file_reader,
            )

        if augmentation:
            self._logger.debug("Init augmentation transforms for time and pitch shift")
            self.t_amplitude = T.RandomAmplitude(3, -6)
            self.t_timestretch = T.RandomTimeStretch()
            self.t_pitchshift = T.RandomPitchSift()
        else:
            self._logger.debug("Running without augmentation")

        if self.freq_compression == "linear":
            self.t_compr_f = T.Interpolate(
                n_freq_bins, sr, f_min, f_max
            )
        elif self.freq_compression == "mel":
            self.t_compr_f = T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max)
        elif self.freq_compression == "mfcc":
            self.t_compr_f = T.Compose(
                T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max)
            )
            self.t_compr_mfcc = T.M2MFCC(n_mfcc=32)
        else:
            raise "Undefined frequency compression"

        if augmentation:
            if noise_files:
                self._logger.debug("Init augmentation transform for random noise addition")
                self.t_addnoise = T.RandomAddNoise(
                    noise_files,
                    self.t_spectrogram,
                    T.Compose(self.t_timestretch, self.t_pitchshift, self.t_compr_f),
                    min_length=seq_len,
                    return_original=True
                )
            else:
                self.t_addnoise = None
                self._logger.debug("No noise augmentation")

        self.t_compr_a = T.Amp2Db(min_level_db=DefaultSpecDatasetOps["min_level_db"])

        if min_max_normalize:
            self.t_norm = T.MinMaxNormalize()
            self._logger.debug("Init min-max-normalization activated")
        else:
            self.t_norm = T.Normalize(
                min_level_db=DefaultSpecDatasetOps["min_level_db"],
                ref_level_db=DefaultSpecDatasetOps["ref_level_db"],
            )
            self._logger.debug("Init 0/1-dB-normalization activated")

        self.t_subseq = T.PaddedSubsequenceSampler(seq_len, dim=1, random=augmentation)

    """
    Computes per filename the entire data preprocessing pipeline containing all transformations and returns the
    preprocessed sample as well as the ground truth label 
    """
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.working_dir is not None:
            file = os.path.join(self.working_dir, file_name)
        else:
            file = file_name
        sample = self.t_spectrogram(file)

        if self.augmentation:
            sample = self.t_amplitude(sample)
            sample = self.t_pitchshift(sample)
            sample = self.t_timestretch(sample)
        sample = self.t_compr_f(sample)

        if (
            self.augmentation and self.t_addnoise is not None
        ):
            sample, ground_truth = self.t_addnoise(sample)
            if self.freq_compression != "mfcc":
                ground_truth = self.t_compr_a(ground_truth)
            else:
                ground_truth = self.t_compr_mfcc(ground_truth)
            ground_truth = self.t_norm(ground_truth)
        else:
            ground_truth = None
        if self.freq_compression != "mfcc":
            sample = self.t_compr_a(sample)
        else:
            sample = self.t_compr_mfcc(sample)
        sample = self.t_norm(sample)
        if ground_truth is not None:
            stacked = torch.cat((sample, ground_truth), dim=0)
            stacked = self.t_subseq(stacked)
            sample = stacked[0].unsqueeze(0)
            ground_truth = stacked[1].unsqueeze(0)
        else:
            sample = self.t_subseq(sample)
        label = self.load_label(file)
        if ground_truth is not None:
            label["ground_truth"] = ground_truth
        return sample, label

    """
    Generate label dict containing filename and whether it is a target signal (call)
    or a noise signal (noise)
    """
    def load_label(self, file_name: str):
        label = dict()
        label["file_name"] = file_name
        label["call"] = self.get_class_type(file_name)
        return label


    def get_class_type_from_idx(self, idx):
        for t, n in self.class_dist_dict.items():
            if n == idx:
                 return t
        raise ValueError("Unkown class type for idx ", idx)

    def get_class_type(self, file_name):
        if platform.system() == "Windows":
            class_name = file_name.split("\\")[-1].split("-", 1)[0]
        else:
            class_name = file_name.split("/")[-1].split("-", 1)[0]
        return self.class_dist_dict[class_name]

"""
Dataset for processing an audio tape via a sliding window approach using a given
sequence length and hop size.
"""
class StridedAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_name,
        sequence_len: int,
        hop: int,
        sr: int = 44100,
        fft_size: int = 4096,
        fft_hop: int = 441,
        n_freq_bins: int = 256,
        freq_compression: str = "linear",
        f_min: int = 200,
        f_max: int = 18000,
        min_max_normalize=False
    ):
        self.sequence_len = sequence_len
        self.hop = hop

        self.audio = T.load_audio_file(file_name, sr=sr, mono=True)
        self.n_frames = self.audio.shape[1]

        self.t = [
            T.PreEmphasize(DefaultSpecDatasetOps["preemphases"]),
            T.Spectrogram(fft_size, fft_hop, center=False),
        ]

        if freq_compression == "linear":
            self.t.append(T.Interpolate(n_freq_bins, sr, f_min, f_max))
        elif freq_compression == "mel":
            self.t.append(T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max))
        elif freq_compression == "mfcc":
            t_mel = T.F2M(sr=sr, n_mels=n_freq_bins, f_min=f_min, f_max=f_max)
            self.t.append(T.Compose(t_mel, T.M2MFCC()))
        else:
            raise "Undefined frequency compression"
        self.t.append(T.Amp2Db(min_level_db=DefaultSpecDatasetOps["min_level_db"]))

        if min_max_normalize:
            self.t.append(T.MinMaxNormalize())
        else:
            self.t.append(
                T.Normalize(
                    min_level_db=DefaultSpecDatasetOps["min_level_db"],
                    ref_level_db=DefaultSpecDatasetOps["ref_level_db"],
                )
            )
        self.t = T.Compose(self.t)


    def __len__(self):
        return max(int(ceil((self.n_frames + 1 - self.sequence_len) / self.hop)), 1)

    """
    Extracts signal part according to the current and respective position of the given audio file.
    """
    def __getitem__(self, idx):
        start = idx * self.hop
        end = min(start + self.sequence_len, self.n_frames)
        y = self.audio[:, start:end]
        return self.t(y)

    def __delete__(self):
        self.loader.join()

    def __exit__(self, *args):
        self.loader.join()
