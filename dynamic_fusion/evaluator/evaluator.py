from collections import defaultdict
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d  # pyright: ignore
from time import time


from dynamic_fusion.evaluator.dataset import CocoTestDataset
from dynamic_fusion.networks.decoding_nets.mlp import MLP
from dynamic_fusion.networks.reconstruction_nets.conv_gru_nets import ConvGruNetV1
from dynamic_fusion.utils.datatypes import Batch
from dynamic_fusion.utils.loss import get_reconstruction_loss
from dynamic_fusion.utils.network import network_data_to_device
from tqdm import tqdm

TRAIN_DATASET_DIRECTORY = Path("data", "interim", "coco", "2subbins")
TEST_DATASET_DIRECTORY = Path("data", "interim", "coco", "test", "2subbins_new")
IMPLICIT = Path("runs/continuous/third_run_with_mean_std_count/latest_checkpoint.pt")
UNFOLDING = Path("runs/continuous/fourth_run_with_unfolding/latest_checkpoint.pt")
EXPLICIT = Path("runs/explicit/latest_checkpoint.pt")

BATCH_SIZE = 1
NUM_WORKERS = 0
USE_MEAN, USE_STD, USE_COUNT = (True, True, True)

ENCODING_INPUT_SIZE = 2
ENCODING_HIDDEN_SIZE = 24
ENCODING_KERNEL_SIZE = 3
ENCODING_OUTPUT_SIZE = 64

DECODING_HIDDEN_SIZE = 128
DECODING_HIDDEN_LAYERS = 4

SKIP_FIRST = 4

LOSS_BATCH_SIZE = 10

DICT_OUTPUT = "./runs/evaluation/new_explicit_loss_dict.pkl"


def load_encoding_network(implicit: bool, checkpoint_path: Path) -> nn.Module:
    total_input_shape = ENCODING_INPUT_SIZE * (1 + USE_MEAN + USE_STD + USE_COUNT)
    output_size = ENCODING_OUTPUT_SIZE if implicit else 1
    encoding_network = ConvGruNetV1(
        input_size=total_input_shape,
        hidden_size=ENCODING_HIDDEN_SIZE,
        out_size=output_size,
        kernel_size=ENCODING_KERNEL_SIZE,
    )
    checkpoint = torch.load(checkpoint_path)  # type: ignore
    # For backward compatibility (key was changed)
    if "encoding_state_dict" in checkpoint.keys():
        encoding_network.load_state_dict(checkpoint["encoding_state_dict"])
    # For compatibility reasons
    elif "reconstruction_state_dict" in checkpoint.keys():
        encoding_network.load_state_dict(checkpoint["reconstruction_state_dict"])

    return encoding_network


def load_decoding_network(feature_unfolding: bool, checkpoint_path: Path) -> nn.Module:
    input_shape = ENCODING_OUTPUT_SIZE
    if feature_unfolding:
        input_shape *= 9
    input_shape += 1

    decoding_network = MLP(
        input_size=input_shape,
        hidden_size=DECODING_HIDDEN_SIZE,
        hidden_layers=DECODING_HIDDEN_LAYERS,
    )

    checkpoint = torch.load(checkpoint_path)  # type: ignore
    decoding_network.load_state_dict(checkpoint["decoding_state_dict"])

    return decoding_network


def evaluate_network(
    investigated_timestamps: Optional[List[float]],
    implicit: bool,
    feature_unfolding: bool,
    test: bool,
    loss_names: List[str],
    interpolation_type: str = "linear",
    interpolate_implicit: bool = False,
) -> Dict[Tuple[float, str], float]:
    if implicit:
        checkpoint = IMPLICIT if not feature_unfolding else UNFOLDING
    else:
        checkpoint = EXPLICIT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_functions = [get_reconstruction_loss(x, device) for x in loss_names]

    dataset = CocoTestDataset(
        TEST_DATASET_DIRECTORY if test else TRAIN_DATASET_DIRECTORY,
        implicit,
        investigated_timestamps,
        data_generator_target_image_size=None if test else (180, 240),
    )
    data_loader: DataLoader[Batch] = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    encoding_network = load_encoding_network(implicit, checkpoint).to(device)
    if implicit:
        decoding_network = load_decoding_network(feature_unfolding, checkpoint).to(device)

    # This is a hack that makes the code a lot cleaner
    if investigated_timestamps is None:
        investigated_timestamps = [1]
    losses_dict: Dict[Tuple[float, str], float] = defaultdict(float)

    losses_added = 0
    for i, batch in enumerate(tqdm(data_loader)):
        batch = network_data_to_device(batch, device, USE_MEAN, USE_STD, USE_COUNT)
        (
            event_polarity_sums,
            timestamp_means,
            timestamp_stds,
            event_counts,
            video,
            continuous_timestamps,
            continuous_timestamp_frames,
        ) = batch

        if not implicit:
            explicit_predictions = []

        encoding_network.reset_states()
        for t in range(event_polarity_sums.shape[1]):
            t0 = time()
            encoding_prediction = encoding_network(
                event_polarity_sums[:, t],
                timestamp_means[:, t] if USE_MEAN else None,
                timestamp_stds[:, t] if USE_STD else None,
                event_counts[:, t] if USE_COUNT else None,
            )
            if not implicit and investigated_timestamps != [1]:
                explicit_predictions.append(encoding_prediction.clone())
                continue

            if t < SKIP_FIRST:
                continue

            if not implicit and investigated_timestamps == [1]:
                for loss_name, loss_function in zip(loss_names, loss_functions):
                    loss = loss_function(encoding_prediction, video[:, t]).mean()
                    losses_dict[(1, loss_name)] += loss.item()
                continue

            # Implicit
            if feature_unfolding:
                unfolded_prediction = torch.nn.functional.unfold(encoding_prediction, kernel_size=3, padding=1, stride=1)
                encoding_prediction = einops.rearrange(
                    unfolded_prediction,
                    "B C (X Y) -> B C X Y",
                    X=continuous_timestamp_frames.shape[4],
                )
            losses_added += 1

            # Non-interpolated implicit
            if not interpolate_implicit:
                for n, investigated_timestamp in enumerate(investigated_timestamps):
                    expanded_timestamps = einops.repeat(
                        continuous_timestamps[:, n, t],
                        "B -> B 1 X Y",
                        X=continuous_timestamp_frames.shape[4],
                        Y=continuous_timestamp_frames.shape[5],
                    )
                    encoding_and_time = torch.concat([encoding_prediction, expanded_timestamps], dim=1)

                    encoding_and_time = einops.rearrange(encoding_and_time, "B C X Y -> B X Y C")
                    decoding_prediction = decoding_network(encoding_and_time)
                    prediction = einops.rearrange(decoding_prediction, "B X Y 1 -> B 1 X Y")

                    for loss_name, loss_function in zip(loss_names, loss_functions):
                        loss = loss_function(prediction, continuous_timestamp_frames[:, n, t]).mean()
                        losses_dict[(investigated_timestamp, loss_name)] += loss.item()

            # Interpolated implicit
            assert torch.all(continuous_timestamps[:, 0, t] == 0)
            assert torch.all(continuous_timestamps[:, -1, t] == 1)
            start_and_end_timestamps = einops.repeat(
                continuous_timestamps[:, [0, -1], t],
                "B 2 -> (B 2) 1 X Y",
                X=continuous_timestamp_frames.shape[4],
                Y=continuous_timestamp_frames.shape[5],
            )

            encoding = einops.repeat(encoding_prediction, "B ... -> (B 2) ...")

            encoding_and_time = torch.concat([encoding, start_and_end_timestamps], dim=1)
            encoding_and_time = einops.rearrange(encoding_and_time, "(B 2) C X Y -> (B 2) X Y C")
            decoding_prediction = decoding_network(encoding_and_time)
            prediction = einops.rearrange(decoding_prediction, "(B 2) X Y 1 -> (B 2) 1 X Y")

        # Explicit
        if not implicit and investigated_timestamps != [1]:
            # interpolate explicit predictions to get continuous time
            predicted_video = torch.stack(explicit_predictions, dim=1).cpu()
            np_prediction: Float[np.ndarray, "B T 1 X Y"] = predicted_video.numpy()
            predicted_video_timestamps = np.arange(1, np_prediction.shape[1] + 1, 1)

            interpolation = interp1d(
                predicted_video_timestamps,
                np_prediction,
                kind=interpolation_type,
                axis=1,
            )

            # This relies on the assumption that
            # continuous_timestamps[i] = continuous_timestamps[j] for all i,j
            # (batch dimension)
            # TMinusOne because we cannot interpolate in [0,1]
            interpolation_timestamps: Float[np.ndarray, "N TMinusOne"] = (
                einops.repeat(predicted_video_timestamps[1:], "TMinusOne -> () TMinusOne") - 1 + continuous_timestamps[0, :, 1:].cpu().numpy()
            )

            flattened_interpolated_predictions = interpolation(einops.rearrange(interpolation_timestamps, "N TMinusOne -> (N TMinusOne)"))

            # The following is true if investigated_timestamps[0] == 0
            # and investigated_timestamps[-1] == 1:
            #   interpolated_predictions[0, -1, 0] == interpolation_timestamps[0,0,1]

            interpolated_predictions = einops.rearrange(
                flattened_interpolated_predictions,
                "B (N TMinusOne) 1 X Y -> B N TMinusOne 1 X Y",
                N=interpolation_timestamps.shape[0],
            )

            interpolated_predictions = torch.tensor(interpolated_predictions).float().to(device)
            losses_added += 1
            for n, investigated_timestamp in enumerate(investigated_timestamps):
                for loss_name, loss_function in zip(loss_names, loss_functions):
                    x = einops.rearrange(
                        interpolated_predictions[:, n, SKIP_FIRST - 1 :],
                        "... 1 X Y -> (...) 1 X Y",
                    )
                    y = einops.rearrange(
                        continuous_timestamp_frames[:, n, SKIP_FIRST:],
                        "... 1 X Y -> (...) 1 X Y",
                    )
                    current_losses = []
                    for i_loss in range(0, x.shape[0], LOSS_BATCH_SIZE):
                        loss = loss_function(
                            x[i_loss : i_loss + LOSS_BATCH_SIZE],
                            y[i_loss : i_loss + LOSS_BATCH_SIZE],
                        ).mean()
                        current_losses.append(loss.item())
                    losses_dict[(investigated_timestamp, loss_name)] += np.mean(current_losses)

    losses_dict = {key: value / losses_added for key, value in losses_dict.items()}
    return losses_dict


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    with torch.no_grad():
        investigated_timestamps = [0, 0.2, 0.4, 0.6, 0.8, 1]
        test = True
        loss_names = ["LPIPS", "L1", "L2"]

        cubic = evaluate_network(investigated_timestamps, False, False, test, loss_names, "cubic")
        linear = evaluate_network(investigated_timestamps, False, False, test, loss_names, "linear")

        implicit = evaluate_network(investigated_timestamps, True, False, test, loss_names)
        unfolded = evaluate_network(investigated_timestamps, True, True, test, loss_names)

        with open(DICT_OUTPUT, "wb") as f:
            pickle.dump({"cubic": cubic, "linear": linear}, f)  # "implicit": implicit, "unfolded": unfolded}, f)
