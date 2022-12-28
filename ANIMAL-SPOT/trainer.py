"""
Module: trainer.py
Authors: Christian Bergler, Hendrik Schroeter
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 27.12.2022
"""

import os
import copy
import math
import time
import operator
import platform
import numpy as np
import utils.metrics as m

import torch
import torch.nn as nn

from typing import Union
from utils.logging import Logger
from utils.aucmeter import AUCMeter
from tensorboardX import SummaryWriter

from utils.checkpoints import CheckpointHandler
from utils.confusionmeter import ConfusionMeter
from utils.early_stopping import EarlyStoppingCriterion
from utils.summary import prepare_img, roc_fig, confusion_matrix_fig

"""
Class which implements network training, validation and testing as well as writing checkpoints, logs, summaries, and saving the final model.
"""
class Trainer:

    """
    Initializing summary writer and checkpoint handler as well as setting required variables for training.
    """
    def __init__(
        self,
        model: nn.Module,
        logger: Logger,
        prefix: str = "",
        checkpoint_dir: Union[str, None] = None,
        summary_dir: Union[str, None] = None,
        n_summaries: int = 4,
        input_shape: tuple = None,
        start_scratch: bool = False,
    ):
        self.model = model
        self.logger = logger
        self.prefix = prefix

        self.logger.info("Init summary writer")

        if summary_dir is not None:
            run_name = prefix + "_" if prefix != "" else ""
            run_name += "{time}-{host}".format(
                time=time.strftime("%y-%m-%d-%H-%M", time.localtime()),
                host=platform.uname()[1],
            )
            summary_dir = os.path.join(summary_dir, run_name)

        self.n_summaries = n_summaries
        self.writer = SummaryWriter(summary_dir)

        if input_shape is not None:
            dummy_input = torch.rand(input_shape)
            self.logger.info("Writing graph to summary")
            self.writer.add_graph(self.model, dummy_input)

        if checkpoint_dir is not None:
            self.cp = CheckpointHandler(
                checkpoint_dir, prefix=prefix, logger=self.logger
            )
        else:
            self.cp = None

        self.start_scratch = start_scratch

        self.class_dist_dict = None

    """
    Starting network training from scratch or loading existing checkpoints. The model training and validation is processed for a given
    number of epochs while storing all relevant information (metrics, summaries, logs, checkpoints) after each epoch. After the training 
    is stopped (either no improvement of the chosen validation metric for a given number of epochs, or maximum training epoch is reached)
    the model will be tested on the independent test set and saved to the selected model target directory.
    """
    def fit(
        self,
        train_loader,
        val_loader,
        test_loader,
        loss_fn,
        optimizer,
        scheduler,
        n_epochs,
        val_interval,
        patience_early_stopping,
        device,
        metrics: Union[list, dict] = [],
        val_metric: Union[int, str] = "loss",
        val_metric_mode: str = "min",
        start_epoch=0,
    ):
        self.logger.info("Init model on device '{}'".format(device))
        self.model = self.model.to(device)
        self.class_dist_dict = train_loader.dataset.class_dist_dict

        best_model = copy.deepcopy(self.model.state_dict())
        best_metric = 0.0 if val_metric_mode == "max" else float("inf")

        patience_stopping = math.ceil(patience_early_stopping / val_interval)
        patience_stopping = int(max(1, patience_stopping))
        early_stopping = EarlyStoppingCriterion(
            mode=val_metric_mode, patience=patience_stopping
        )

        if not self.start_scratch and self.cp is not None:
            checkpoint = self.cp.read_latest()
            if checkpoint is not None:
                try:
                    try:
                        self.model.load_state_dict(checkpoint["modelState"])
                    except RuntimeError as e:
                        self.logger.error(
                            "Failed to restore checkpoint: "
                            "Checkpoint has different parameters"
                        )
                        self.logger.error(e)
                        raise SystemExit

                    optimizer.load_state_dict(checkpoint["trainState"]["optState"])
                    start_epoch = checkpoint["trainState"]["epoch"] + 1
                    best_metric = checkpoint["trainState"]["best_metric"]
                    best_model = checkpoint["trainState"]["best_model"]
                    early_stopping.load_state_dict(
                        checkpoint["trainState"]["earlyStopping"]
                    )
                    scheduler.load_state_dict(checkpoint["trainState"]["scheduler"])
                    self.logger.info("Resuming with epoch {}".format(start_epoch))
                except KeyError:
                    self.logger.error("Failed to restore checkpoint")
                    raise

        since = time.time()

        self.logger.info("Class Distribution: " + str(self.class_dist_dict))

        self.logger.info("Start training model " + self.prefix)

        try:
            if val_metric_mode == "min":
                val_comp = operator.lt
            else:
                val_comp = operator.gt
            for epoch in range(start_epoch, n_epochs):
                self.train_epoch(
                    epoch, train_loader, loss_fn, optimizer, metrics, device
                )
                if epoch % val_interval == 0 or epoch == n_epochs - 1:
                    val_loss = self.test_epoch(
                        epoch, val_loader, loss_fn, metrics, device, phase="val"
                    )
                    if val_metric == "loss":
                        val_result = val_loss
                    else:
                        val_result = metrics[val_metric].get()
                    if val_comp(val_result, best_metric):
                        best_metric = val_result
                        best_model = copy.deepcopy(self.model.state_dict())
                    self.cp.write(
                        {
                            "modelState": self.model.state_dict(),
                            "trainState": {
                            "epoch": epoch,
                            "best_metric": best_metric,
                            "best_model": best_model,
                            "optState": optimizer.state_dict(),
                            "earlyStopping": early_stopping.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            },
                            "classes": train_loader.dataset.class_dist_dict
                        }
                    )
                    scheduler.step(val_result)
                    if early_stopping.step(val_result):
                        self.logger.info(
                            "No improvment over the last {} epochs. Stopping.".format(
                                patience_early_stopping
                            )
                        )
                        break
        except Exception:
            import traceback
            self.logger.warning(traceback.format_exc())
            self.logger.warning("Aborting...")
            self.logger.close()
            raise SystemExit

        self.model.load_state_dict(best_model)
        final_loss = self.test_epoch(
            0, test_loader, loss_fn, metrics, device, phase="test"
        )
        if val_metric == "loss":
            final_metric = final_loss
        else:
            final_metric = metrics[val_metric].get()

        time_elapsed = time.time() - since
        self.logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        self.logger.info("Best val metric: {:4f}".format(best_metric))
        self.logger.info("Final test metric: {:4f}".format(final_metric))

        return self.model

    """
    Training of one epoch using pre-extracted training data, loss function, optimizer, and respective metrics
    """
    def train_epoch(self, epoch, train_loader, loss_fn, optimizer, metrics, device):
        self.logger.debug("train|{}|start".format(epoch))
        if isinstance(metrics, list):
            for metric in metrics:
                metric.reset(device)
        else:
            for metric in metrics.values():
                metric.reset(device)

        self.model.train()

        epoch_start = time.time()
        start_data_loading = epoch_start
        data_loading_time = m.Sum(torch.device("cpu"))
        epoch_loss = m.Mean(device)

        if train_loader.dataset.num_classes == 2:
            auc = AUCMeter()
            confusion = None
        else:
            auc = None
            confusion = ConfusionMeter(n_categories=train_loader.dataset.num_classes)

        for i, (features, label) in enumerate(train_loader):
            features = features.to(device)
            call_label = None

            if "call" in label:
                call_label = label["call"].to(device, non_blocking=True, dtype=torch.int64)

            data_loading_time.update(torch.Tensor([(time.time() - start_data_loading)]))
            optimizer.zero_grad()

            output = self.model(features)

            loss = loss_fn(output, call_label)
            loss.backward()

            optimizer.step()

            epoch_loss.update(loss)

            prediction = None

            if call_label is not None:
                prediction = torch.argmax(output.data, dim=1)
                if isinstance(metrics, list):
                    for metric in metrics:
                        metric.update(call_label, prediction)
                else:
                    for metric in metrics.values():
                        metric.update(call_label, prediction)

                if auc is not None:
                    score = nn.functional.softmax(output, dim=1)[:, 1]
                    auc.add(score.detach(), call_label)
                if confusion is not None:
                    confusion.add(prediction, call_label)

            if i == 0:
                self.write_summaries(
                    features=features,
                    labels=call_label,
                    prediction=prediction,
                    file_names=label["file_name"],
                    epoch=epoch,
                    phase="train"
                )
            start_data_loading = time.time()

        self.write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            metrics=metrics,
            lr=optimizer.param_groups[0]["lr"],
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase="train",
        )

        if call_label is not None:
            if auc is not None:
                self.write_roc_curve_summary(*auc.value(), epoch, phase="train")
            if confusion is not None:
                confusion_matrix_raw = confusion.confusion.clone()
                confusion_matrix_norm = confusion.value()
                label_str = [train_loader.dataset.get_class_type_from_idx(i) for i in range(confusion_matrix_norm.shape[0])]
                self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase="train", norm=True, numbering=True)
                self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase="train", norm=True, numbering=False)
                self.write_confusion_summary(confusion_matrix_raw, label_str, epoch=epoch, phase="train", norm=False, numbering=True)

        self.writer.flush()

        return epoch_loss.get()

    """ 
    Validation/Testing using pre-extracted validation/test data, given loss function and respective metrics.
    The parameter 'phase' is used to switch between validation and test
    """
    def test_epoch(self, epoch, test_loader, loss_fn, metrics, device, phase="val"):
        self.logger.debug("{}|{}|start".format(phase, epoch))
        self.model.eval()

        with torch.no_grad():
            if isinstance(metrics, list):
                for metric in metrics:
                    metric.reset(device)
            else:
                for metric in metrics.values():
                    metric.reset(device)
            epoch_start = time.time()
            start_data_loading = epoch_start
            data_loading_time = m.Sum(torch.device("cpu"))

            epoch_loss = m.Mean(device)

            if test_loader.dataset.num_classes == 2:
                auc = AUCMeter()
                confusion = None
            else:
                auc = None
                confusion = ConfusionMeter(n_categories=test_loader.dataset.num_classes)

            for i, (features, label) in enumerate(test_loader):
                features = features.to(device)
                call_label = None

                if "call" in label:
                    call_label = label["call"].to(device, non_blocking=True, dtype=torch.int64)

                data_loading_time.update(
                    torch.Tensor([(time.time() - start_data_loading)])
                )

                output = self.model(features)

                loss = loss_fn(output, call_label)
                epoch_loss.update(loss)

                prediction = None

                if call_label is not None:
                    prediction = torch.argmax(output.data, dim=1)
                    if isinstance(metrics, list):
                        for metric in metrics:
                            metric.update(call_label, prediction)
                    else:
                        for metric in metrics.values():
                            metric.update(call_label, prediction)

                    if auc is not None:
                        score = nn.functional.softmax(output, dim=1)[:, 1]
                        auc.add(score, call_label)
                    if confusion is not None:
                        confusion.add(prediction, call_label)

                if i == 0:
                    self.write_summaries(
                        features=features,
                        labels=call_label,
                        prediction=prediction,
                        file_names=label["file_name"],
                        epoch=epoch,
                        phase=phase
                    )
                start_data_loading = time.time()

        self.write_scalar_summaries_logs(
            loss=epoch_loss.get(),
            metrics=metrics,
            epoch_time=time.time() - epoch_start,
            data_loading_time=data_loading_time.get(),
            epoch=epoch,
            phase=phase,
        )

        if call_label is not None:
            if auc is not None:
                self.write_roc_curve_summary(*auc.value(), epoch, phase=phase)
            if confusion is not None:
                confusion_matrix_raw = confusion.confusion.clone()
                confusion_matrix_norm = confusion.value()
                label_str = [test_loader.dataset.get_class_type_from_idx(i) for i in range(confusion_matrix_norm.shape[0])]
                self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase=phase, norm=True, numbering=True)
                self.write_confusion_summary(confusion_matrix_norm, label_str, epoch=epoch, phase=phase, norm=True, numbering=False)
                self.write_confusion_summary(confusion_matrix_raw, label_str, epoch=epoch, phase=phase, norm=False, numbering=True)

        self.writer.flush()

        return epoch_loss.get()

    """
    Writes image summary per partition (spectrograms and the corresponding predictions)
    """
    def write_summaries(
        self,
        features,
        labels=None,
        prediction=None,
        file_names=None,
        epoch=None,
        phase="train"
    ):
        with torch.no_grad():
            self.write_img_summaries(
                features,
                labels=labels,
                prediction=prediction,
                file_names=file_names,
                epoch=epoch,
                phase=phase
            )

    """
    Writes image summary per partition with respect to the prediction output (true predictions - true positive/negative, false 
    predictions - false positive/negative)
    """
    def write_img_summaries(
        self,
        features,
        labels=None,
        prediction=None,
        file_names=None,
        epoch=None,
        phase="train"
    ):
        with torch.no_grad():
            if file_names is not None:
                if isinstance(file_names, torch.Tensor):
                    file_names = file_names.cpu().numpy()
                elif isinstance(file_names, list):
                    file_names = np.asarray(file_names)
            if labels is not None and prediction is not None:
                features = features.cpu()
                labels = labels.cpu()
                prediction = prediction.cpu()
                t_i = torch.eq(prediction, labels)

                for idx in range(len(t_i)):
                    if t_i[idx]:
                        name_t = "true - " + str(self.get_class_type_from_idx(labels[idx].item())) + " as " + str(self.get_class_type_from_idx(prediction[idx]))
                    else:
                        name_t = "false - " + str(self.get_class_type_from_idx(labels[idx].item())) + " as " + str(self.get_class_type_from_idx(prediction[idx]))

                    try:
                        self.writer.add_image(
                            tag=phase + "/" + name_t,
                            img_tensor=prepare_img(
                                features[idx].unsqueeze(dim=0),
                                num_images=self.n_summaries,
                                file_names=file_names[idx],
                            ),
                            global_step=epoch,
                        )
                    except ValueError:
                        pass
            else:
                self.writer.add_image(
                    tag=phase + "/input",
                    img_tensor=prepare_img(features, num_images=self.n_summaries, file_names=file_names),
                    global_step=epoch,
                )

    """
    Writes scalar summary per partition including loss, confusion matrix, accuracy, recall, f1-score, true positive rate,
    false positive rate, precision, data_loading_time, epoch time
    """
    def write_scalar_summaries_logs(
        self,
        loss: float,
        metrics: Union[list, dict] = [],
        lr: float = None,
        epoch_time: float = None,
        data_loading_time: float = None,
        epoch=None,
        phase="train",
    ):
        with torch.no_grad():
            log_str = phase
            if epoch is not None:
                log_str += "|{}".format(epoch)
            self.writer.add_scalar(phase + "/epoch_loss", loss, epoch)
            log_str += "|loss:{:0.3f}".format(loss)
            if isinstance(metrics, dict):
                for name, metric in metrics.items():
                    self.writer.add_scalar(phase + "/" + name, metric.get(), epoch)
                    log_str += "|{}:{:0.3f}".format(name, metric.get())
            else:
                for i, metric in enumerate(metrics):
                    self.writer.add_scalar(
                        phase + "/metric_" + str(i), metric.get(), epoch
                    )
                    log_str += "|m_{}:{:0.3f}".format(i, metric.get())
            if lr is not None:
                self.writer.add_scalar("lr", lr, epoch)
                log_str += "|lr:{:0.2e}".format(lr)
            if epoch_time is not None:
                self.writer.add_scalar(phase + "/time", epoch_time, epoch)
                log_str += "|t:{:0.1f}".format(epoch_time)
            if data_loading_time is not None:
                self.writer.add_scalar(
                    phase + "/data_loading_time", data_loading_time, epoch
                )
            self.logger.info(log_str)

    """
    Writes roc curve summary for validation and test set 
    """
    def write_roc_curve_summary(self, auc, tpr, fpr, epoch=None, phase=""):
        with torch.no_grad():
            if phase != "":
                phase += "_"
            fig = roc_fig(tpr, fpr, auc)
            self.writer.add_figure(phase + "roc/roc", fig, epoch)

    """
    Writes confusion matrix summary for validation and test set 
    """
    def write_confusion_summary(self, confusion_matrix, label_str=None, epoch=None, phase="", norm=True, numbering=True):
        with torch.no_grad():
            if phase != "":
                phase += "_"
            fig = confusion_matrix_fig(confusion_matrix, label_str=label_str, numbering=numbering)
            if norm:
                if numbering:
                    cm_file = "confusion_matrix_norm/cm_numbered"
                else:
                    cm_file = "confusion_matrix_norm/cm"
            else:
                if numbering:
                    cm_file = "confusion_matrix_raw/cm_numbered"
                else:
                    cm_file = "confusion_matrix_raw/cm"
            self.writer.add_figure(phase + cm_file, fig, epoch)


    def get_class_type_from_idx(self, idx):
        for t, n in self.class_dist_dict.items():
            if n == idx:
                 return t
        raise ValueError("Unkown class type for idx ", idx)
