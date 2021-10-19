import sys
from collections import deque
from enum import Enum, auto

import cv2
import numpy as np
from tqdm import tqdm


class Methods(Enum):
    DIS = auto()
    Farneback = auto()


METHOD = Methods.DIS
RESIZE_DIM = 1280
MOTION_MUL_DISP = 2.

WIN_NAME = "Dense optical flow"

TRC_T = 0.005
TRC_V_T = 150


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == "__main__":
    # Get a VideoCapture object from video and store it in vs
    try:
        print("Processing '%s'" % sys.argv[1])
    except IndexError:
        raise ValueError("Path to video is not provided!")
    vc = cv2.VideoCapture(sys.argv[1])
    if not vc.isOpened():
        raise IOError("cannot open video")
    # Read first frame
    ret, first_frame = vc.read()
    # Scale and resize image
    orig_h, orig_w = first_frame.shape[:2]
    max_dim = max(first_frame.shape)
    scale = RESIZE_DIM/max_dim
    out_h, out_w = int(scale*orig_h), int(scale*orig_w)
    first_frame = cv2.resize(
        first_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    # Convert to gray scale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Create mask
    mask = np.zeros_like(first_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255

    out = cv2.VideoWriter(
        'preview.mp4', cv2.VideoWriter_fourcc(*"mp4v"), vc.get(cv2.CAP_PROP_FPS), (out_w, out_h))
    mask_out = cv2.VideoWriter(
        'mask_out.mp4', cv2.VideoWriter_fourcc(*"mp4v"),
        vc.get(cv2.CAP_PROP_FPS), (orig_w, orig_h))
    dq = deque(maxlen=3)

    if METHOD == Methods.DIS:
        inst = cv2.DISOpticalFlow.create(cv2.DISOpticalFlow_PRESET_MEDIUM)
        use_temporal_propagation = True

    cv2.namedWindow(
        WIN_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    flow = None
    pbar = tqdm(total=int(vc.get(cv2.CAP_PROP_FRAME_COUNT)))
    try:
        while True:
            # Read a frame from video
            ret, frame_orig = vc.read()
            if frame_orig is None:
                break
            pbar.update(1)
            # Convert new frame format`s to gray scale and resize gray frame obtained
            gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(
                gray, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if METHOD == Methods.DIS:
                # Calculate dense optical flow by DIS method
                if flow is not None and use_temporal_propagation:
                    # warp previous flow to get an initial approximation for the current flow:
                    flow = inst.calc(prev_gray, gray, warp_flow(flow, flow))
                else:
                    flow = inst.calc(prev_gray, gray, None)
            elif METHOD == Methods.Farneback:
                # Calculate dense optical flow by Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, pyr_scale=0.5,
                    levels=5, winsize=19, iterations=5, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            # Compute the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hist, bins = np.histogram(
                np.clip(magnitude, 0, 255).flatten(),
                256, [0, 256])
            trc = np.sum(hist[TRC_V_T:]) / (
                magnitude.shape[0] * magnitude.shape[1])
            # Set image hue according to the optical flow direction
            mask[..., 0] = np.rad2deg(angle) / 2
            # Set image value according to the optical flow magnitude (normalized)
            # mask[..., 2] = cv2.normalize(
            #     magnitude, None, 0, 255, cv2.NORM_MINMAX)
            mask[..., 2] = np.clip(
                magnitude * MOTION_MUL_DISP, 0, 255).astype(np.uint8)
            # Convert HSV to RGB (BGR) color representation
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            rgb_upscaled = cv2.resize(
                rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            # Resize frame size to match dimensions
            frame = cv2.resize(
                frame_orig, (out_w, out_h), interpolation=cv2.INTER_AREA)
            # Update previous frame
            prev_gray = gray
            # Open a new window and displays the output frame
            dense_flow = cv2.addWeighted(frame, 0.2, rgb, 0.8, 0)
            dq.append((dense_flow, rgb_upscaled, trc, frame))
            if len(dq) >= 3 and (trc < TRC_T and dq[1][2] >= TRC_T and dq[0][2] < TRC_T):
                frame = dq[1][3]
                dense_flow = cv2.addWeighted(
                    frame, 0.2, np.zeros_like(frame), 0.8, 0)
                cv2.putText(
                    dense_flow, "S (%.4f)" % (trc), (5, magnitude.shape[0] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                dq[1] = (
                    dense_flow, np.zeros((orig_w, orig_h), dtype=np.uint8), dq[1][2], frame)
            if len(dq) >= 3:
                dense_flow, rgb_upscaled, _, _ = dq.popleft()
                out.write(dense_flow)
                mask_out.write(rgb_upscaled)
                # cv2.imshow(WIN_NAME, dense_flow)
            # Frame are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally:
        # The following frees up resources and closes all windows
        pbar.close()
        mask_out.release()
        out.release()
        vc.release()

        cv2.destroyAllWindows()
