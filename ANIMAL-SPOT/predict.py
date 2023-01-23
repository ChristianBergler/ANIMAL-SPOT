#!/usr/bin/env python3

"""
Module: predict.py
Authors: Christian Bergler, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import os
import pickle
import platform
import argparse

import torch
import torch.nn as nn

from pathlib import Path
from math import ceil, floor
from collections import OrderedDict
from utils.summary import plot_spectrogram

from utils.logging import PredictionLogger

from models.classifier import Classifier
from models.residual_encoder import ResidualEncoder as Encoder
from data.audiodataset import StridedAudioDataset, DefaultSpecDatasetOps

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Print additional training and model information.",
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to a model.",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint. "
    "If provided the checkpoint will be used instead of the model.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--output_dir", type=str, default=None, help="The directory to store the output."
)

parser.add_argument(
    "--sequence_len", type=float, default=2, help="Sequence length in [s]."
)

parser.add_argument(
    "--hop", type=float, default=1, help="Hop [s] of subsequent sequences."
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="Threshold for the probability for detecting an orca.",
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--visualize",
    dest="visualize",
    action="store_true",
    help="Additional visualization of the classfied spectrogram",
)

parser.add_argument(
    "--jit_load",
    dest="jit_load",
    action="store_true",
    help="Load model via torch jit (otherwise via torch load).",
)

parser.add_argument(
    "--min_max_norm",
    dest="min_max_norm",
    action="store_true",
    help="activates min-max normalization instead of default 0/1-dB-normalization.",
)

parser.add_argument(
    "--latent_extract",
    dest="latent_extract",
    action="store_true",
    help="Additional extraction of hidden layer information (one layer before final output layer)",
)

parser.add_argument(
    "--input_file", type=str, default=None, help="Single audio file or audio folder including auido files to predict"
)

"""
save the extracted features 
"""
def save_pickle(path, features, name=None):
    if not os.path.isdir(path):
        os.makedirs(path)

    if name is None:
        with open(path + "/features.p", "wb") as f:
            pickle.dump(features, f)
    else:
        with open(path + "/" + name + ".p", "wb") as f:
            pickle.dump(features, f)

"""
load the extracted features 
"""
def load_pickle(path):
    if path.endswith(".p"):
        with open(path, "rb") as f:
            features = pickle.load(f)
    else:
        with open(path+"/features.p", "rb") as f:
            features = pickle.load(f)

    return features

ARGS = parser.parse_args()

log = PredictionLogger("PREDICT", ARGS.debug, ARGS.log_dir)

models = {"encoder": 1, "classifier": 2}

"""
Main function to compute prediction (segmentation) by using a trained model together with a given audio tape by processing a sliding window approach
"""
if __name__ == "__main__":
    if ARGS.jit_load:
        extra_files = {'dataOpts': '', 'classes': ''}
        model = torch.jit.load(ARGS.model_path, _extra_files=extra_files)
        modelState = model.state_dict()
        dataOpts = eval(extra_files['dataOpts'])
        class_dist_dict = eval(extra_files['classes'])
        log.debug("Model successfully load via torch jit: " + str(ARGS.model_path))
    else:
        model_dict = torch.load(ARGS.model_path)
        encoder = Encoder(model_dict["encoderOpts"])
        encoder.load_state_dict(model_dict["encoderState"])
        classifier = Classifier(model_dict["classifierOpts"])
        classifier.load_state_dict(model_dict["classifierState"])
        if model_dict.get("classes") is not None:
            class_dist_dict = model_dict["classes"]
        else:
            class_dist_dict = {"noise": 0, "target": 1}
        model = nn.Sequential(
            OrderedDict([("encoder", encoder), ("classifier", classifier)])
        )
        dataOpts = model_dict["dataOpts"]
        log.debug("Model successfully load via torch load: " + str(ARGS.model_path))

    log.info(model)

    num_classes = len(class_dist_dict)

    if torch.cuda.is_available() and ARGS.cuda:
        model = model.cuda()
    model.eval()

    sr = dataOpts['sr']
    hop_length = dataOpts["hop_length"]
    n_fft = dataOpts["n_fft"]

    try:
        n_freq_bins = dataOpts["num_mels"]
    except KeyError:
        n_freq_bins = dataOpts["n_freq_bins"]

    fmin = dataOpts["fmin"]
    fmax = dataOpts["fmax"]
    freq_cmpr = dataOpts["freq_compression"]
    DefaultSpecDatasetOps["min_level_db"] = dataOpts["min_level_db"]
    DefaultSpecDatasetOps["ref_level_db"] = dataOpts["ref_level_db"]

    log.debug("dataOpts: " + str(dataOpts))
    log.debug("prediction length: " + str(ARGS.sequence_len) + "[s], hop length: " + str(ARGS.hop) + "[s], threshold: " + str(ARGS.threshold) + ", prediction fmin: " + str(fmin) + "[Hz], prediction fmax: " + str(fmax) + "[Hz]")
    sequence_len = int(ceil(ARGS.sequence_len * sr))
    hop = int(ceil(ARGS.hop * sr))

    if ARGS.input_file is None:
        raise Exception("no audio file or directory of audio files have been specified!")

    def get_class_type_from_idx(idx):
        for t, n in class_dist_dict.items():
            if n == idx:
                return t
        raise ValueError("Unkown class type for idx ", idx)

    if os.path.isdir(ARGS.input_file):
        audio_folder = Path(ARGS.input_file)
        audio_files = [f for f in os.listdir(ARGS.input_file) if f.endswith(".wav")]
    else:
        audio_folder = None
        audio_files = [ARGS.input_file]

    log.info("Class Distribution: " + str(class_dist_dict))
    log.info("Predicting files names: {}".format((audio_files)))
    log.info("Predicting {} files".format(len(audio_files)))

    log.close()

    if ARGS.latent_extract:

        file_names = []
        feature_array = []
        spectra_input = []
        features = {
            'filenames': file_names,
            'features': feature_array,
            'spectra_input': spectra_input,
        }

    for file_name in audio_files:
        if platform.system() == "Windows":
            if audio_folder is not None:
                file_name = str(audio_folder) + "\\" + str(file_name)
            file_log = PredictionLogger(file_name.split("\\")[-1].split(".")[0].strip() + "_predict_output", ARGS.debug, ARGS.output_dir)
        else:
            if audio_folder is not None:
                file_name = str(audio_folder) + "/" + str(file_name)
            file_log = PredictionLogger(file_name.split("/")[-1].split(".")[0].strip() + "_predict_output", ARGS.debug, ARGS.output_dir)

        file_log.info(file_name)
        file_log.info("Model Path: " + str(ARGS.model_path))
        file_log.info("dataOpts: " + str(dataOpts))
        file_log.info("prediction length: " + str(ARGS.sequence_len) + "[s], hop length: " + str(
            ARGS.hop) + "[s], threshold: " + str(ARGS.threshold) + ", prediction fmin: " + str(
            fmin) + "[Hz] prediction fmax: " + str(fmax) + "[Hz]")
        file_log.info("class distribution: " + str(class_dist_dict))

        if ARGS.min_max_norm:
            file_log.info("Init min-max-normalization activated")
        else:
            file_log.info("Init 0/1-dB-normalization activated")

        dataset = StridedAudioDataset(
            file_name.strip(),
            sequence_len=sequence_len,
            hop=hop,
            sr=sr,
            fft_size=n_fft,
            fft_hop=hop_length,
            n_freq_bins=n_freq_bins,
            f_min=fmin,
            f_max=fmax,
            freq_compression=freq_cmpr,
            min_max_normalize=ARGS.min_max_norm
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=ARGS.batch_size,
            num_workers=ARGS.num_workers,
            pin_memory=True,
        )

        file_log.info("size of the file(samples)={}".format(dataset.n_frames))
        file_log.info("size of hop(samples)={}".format(hop))
        stop = int(max(floor(dataset.n_frames / hop), 1))
        file_log.info("stop time={}".format(stop))

        with torch.no_grad():
            for i, input in enumerate(data_loader):
                if torch.cuda.is_available() and ARGS.cuda:
                    input = input.cuda()

                out = model(input).cpu()

                if ARGS.latent_extract:

                    feature_array.append(model.classifier.get_layer_output()["hidden_layer_1"].cpu().detach().squeeze().numpy())
                    spectra_input.append(input)

                for n in range(out.shape[0]):
                    t_start = (i * ARGS.batch_size + n) * hop
                    t_end = min(t_start + sequence_len - 1, dataset.n_frames - 1)
                    file_log.debug("start extract={}".format(t_start))
                    file_log.debug("end extract={}".format(t_end))

                    if ARGS.latent_extract:

                        file_names.append(file_name.replace(".wav", "")+"_"+str(round(t_start / sr, 3))+"_"+str(round(t_end / sr, 3))+".wav")

                    if num_classes == 2:
                        prob = torch.nn.functional.softmax(out, dim=1).numpy()[n, 1]
                        pred = int(prob >= ARGS.threshold)
                        detected_class_lbl = get_class_type_from_idx(pred)
                        file_log.info(
                            "time={}-{}, pred={}, prob={}".format(
                                round(t_start / sr, 3), round(t_end / sr, 3), pred, prob
                            )
                        )
                    elif num_classes > 2:
                        prob = torch.nn.functional.softmax(out, dim=1)
                        output = "\n"
                        for index in range(prob.size()[1]):
                            output += str(get_class_type_from_idx(index)) + "=" + str(float(prob[0, index])) + ";\n"

                        result_class_prob, reslut_class_idx = torch.max(prob, 1)
                        detected_class_lbl = get_class_type_from_idx(reslut_class_idx)
                        detected_class_lbl_prob = result_class_prob[0]

                        pred = int(detected_class_lbl_prob >= ARGS.threshold)

                        file_log.info(
                            "time={}-{}, pred={}, pred_class={}-{}, prob={}\noutput_layer:{}".format(
                                round(t_start / sr, 3), round(t_end / sr, 3), pred, detected_class_lbl, reslut_class_idx.item(),
                                detected_class_lbl_prob, output
                            )
                        )

                    else:
                        raise Exception("not a valid number of classes!")

                if ARGS.visualize:
                    plot_spectrogram(spectrogram=input.cpu().squeeze(dim=0).squeeze(dim=0), title="Spectrogram",
                                     output_filepath=ARGS.output_dir + "/net_input_spec_" + str(i) + "_" + str(detected_class_lbl) + "_" +
                                                     file_name.split("/")[-1].split(".")[0] + ".pdf",
                                     sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False,
                                     ax_title="spectrogram")

    if ARGS.latent_extract:
        save_pickle(ARGS.output_dir, features, "animal-spot-classifier")
        file_log.debug("Successfully saved latent classification features (final hidden layer)!")

    file_log.debug("Finished proccessing")

    file_log.close()
