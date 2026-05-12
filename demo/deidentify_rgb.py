# Copyright (c) OpenMMLab. All rights reserved.
"""De-identify an MKV by drawing black eye boxes on the RGB stream.

This script is intentionally separate from ``inferencer_demo.py`` because it
has a stricter video I/O requirement: the de-identified RGB stream is encoded
losslessly, while non-RGB streams can be copied unchanged into a final MKV.

Example:
    PYTHONPATH=. python demo/deidentify_rgb.py input.mkv \
        --output-rgb /tmp/input_rgb_deidentified.mkv \
        --output input_deidentified.mkv \
        --pose2d wholebody --device cpu

	#### EXAMPLE ####        
	python demo/deidentify_rgb.py \
		25235-91971ed1-2201111100_A.mkv \
		--predictions-json 25235-91971ed1-2201111100_A_wholebody.json \
		--output-rgb deidentified_rgb_only.mkv \
		--output deidentified_full.mkv \
		--rgb-stream v:0 \
		--eye-box-size 8 \
		--allow-missing-predictions
        
"""

import argparse
import json
import os
import subprocess
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input MKV path.')
    parser.add_argument(
        '--output-rgb',
        required=True,
        help='Lossless RGB-only MKV with black eye boxes.')
    parser.add_argument(
        '--output',
        default=None,
        help='Optional final MKV. The de-identified RGB stream is combined '
        'with untouched non-RGB streams from the input.')
    parser.add_argument(
        '--rgb-stream',
        default='v:0',
        help='FFmpeg stream specifier for the RGB stream. Use v:0 for the '
        'first video stream. Defaults to v:0.')
    parser.add_argument(
        '--predictions-json',
        default=None,
        help='Optional JSON file with per-frame keypoint predictions. If '
        'provided, MMPoseInferencer is not run.')
    parser.add_argument(
        '--allow-missing-predictions',
        action='store_true',
        help='Allow frames missing from --predictions-json to pass through '
        'without redaction. By default, missing frames raise an error.')
    parser.add_argument(
        '--pose2d',
        default='wholebody',
        help='MMPose 2D pose model alias/config. Ignored when '
        '--predictions-json is set. Defaults to wholebody.')
    parser.add_argument('--pose2d-weights', default=None)
    parser.add_argument('--det-model', default=None)
    parser.add_argument('--det-weights', default=None)
    parser.add_argument(
        '--det-cat-ids', type=int, nargs='+', default=0, help='Detector class.')
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--eye-ids',
        type=int,
        nargs=2,
        default=[1, 2],
        help='Left/right eye keypoint indices. COCO/wholebody eyes are 1 2.')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Minimum keypoint score for drawing an eye box.')
    parser.add_argument(
        '--eye-box-size',
        type=int,
        default=8,
        help='Fixed side length, in pixels, for each black eye square.')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='Optional frame limit for testing. Negative means all frames.')
    parser.add_argument(
        '--ffmpeg-loglevel',
        default='warning',
        help='FFmpeg log level. Defaults to warning.')
    return parser.parse_args()


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def normalize_stream_spec(stream_spec: str) -> str:
    """Return a stream specifier without a leading input-file index."""
    if stream_spec.startswith('0:'):
        return stream_spec[2:]
    return stream_spec


def ffprobe_stream(input_file: str, stream_spec: str) -> dict:
    spec = normalize_stream_spec(stream_spec)
    command = [
        'ffprobe', '-v', 'error', '-select_streams', spec, '-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate', '-of', 'json',
        input_file
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f'ffprobe failed:\n{result.stderr}')

    streams = json.loads(result.stdout).get('streams', [])
    if not streams:
        raise RuntimeError(f'No stream matched {stream_spec!r}.')
    return streams[0]


def parse_fps(stream_info: dict) -> float:
    for key in ('avg_frame_rate', 'r_frame_rate'):
        value = stream_info.get(key)
        if value and value != '0/0':
            fps = float(Fraction(value))
            if fps > 0:
                return fps
    raise RuntimeError('Could not determine input stream FPS.')


def start_decoder(input_file: str, stream_spec: str,
                  loglevel: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            'ffmpeg', '-loglevel', loglevel, '-analyzeduration', '100M',
            '-probesize', '100M', '-i', input_file, '-map',
            f'0:{normalize_stream_spec(stream_spec)}', '-map', '-0:t?',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'
        ],
        stdout=subprocess.PIPE)


def start_lossless_encoder(output_rgb: str, width: int, height: int,
                           fps: float, loglevel: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(os.path.abspath(output_rgb)), exist_ok=True)
    return subprocess.Popen(
        [
            'ffmpeg', '-y', '-loglevel', loglevel, '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
            '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-level', '3', '-g', '1',
            output_rgb
        ],
        stdin=subprocess.PIPE)


def read_frame(decoder: subprocess.Popen, frame_size: int,
               shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    raw = decoder.stdout.read(frame_size)
    if not raw:
        return None
    if len(raw) != frame_size:
        raise RuntimeError(
            f'Incomplete frame from decoder: {len(raw)} of {frame_size} bytes.')
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def load_predictions(predictions_json: str) -> dict:
    """Load MMPose-style video predictions keyed by frame id."""
    with open(predictions_json, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'predictions' in data:
        data = data['predictions']

    predictions = {}
    if isinstance(data, list):
        if all(isinstance(item, dict) and 'frame_id' in item for item in data):
            for item in data:
                predictions[int(item['frame_id'])] = item.get('instances', [])
        elif all(isinstance(item, list) for item in data):
            predictions = {frame_id: item for frame_id, item in enumerate(data)}
        elif all(isinstance(item, dict) and 'keypoints' in item
                 for item in data):
            predictions[0] = data
        else:
            raise ValueError(
                'Unsupported predictions JSON list format. Expected a list of '
                '{"frame_id": int, "instances": [...]}, a list of per-frame '
                'instance lists, or a single-frame instance list.')
    elif isinstance(data, dict):
        for frame_id, instances in data.items():
            predictions[int(frame_id)] = instances
    else:
        raise ValueError('Unsupported predictions JSON format.')

    return predictions


def get_pose_instances(inferencer, frame: np.ndarray, args) -> List[dict]:
    result = next(
        inferencer(
            frame,
            return_datasamples=True,
            bbox_thr=0.3,
            nms_thr=0.3,
            kpt_thr=args.kpt_thr,
            show=False,
            return_vis=False))
    data_sample = result['predictions'][0]
    pred_instances = data_sample.pred_instances.cpu().numpy()
    if 'keypoints' not in pred_instances:
        return []

    keypoints = pred_instances.keypoints
    scores = getattr(pred_instances, 'keypoint_scores', None)
    instances = []
    for instance_id, kpts in enumerate(keypoints):
        instance = {'keypoints': kpts}
        if scores is not None:
            instance['keypoint_scores'] = scores[instance_id]
        instances.append(instance)
    return instances


def get_json_instances(predictions: dict, frame_id: int, args) -> List[dict]:
    if frame_id in predictions:
        return predictions[frame_id]
    if args.allow_missing_predictions:
        return []
    raise RuntimeError(
        f'No predictions found for frame {frame_id}. Re-run with '
        '--allow-missing-predictions to pass missing frames through.')


def instance_keypoints_and_scores(instance: dict) -> Tuple[np.ndarray,
                                                          Optional[np.ndarray]]:
    keypoints = np.asarray(instance.get('keypoints'), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError('Each instance must contain keypoints shaped Kx2 '
                         'or Kx3.')

    scores = None
    for key in ('keypoint_scores', 'keypoints_visible', 'keypoint_score'):
        if key in instance and instance[key] is not None:
            scores = np.asarray(instance[key], dtype=np.float32)
            break
    if scores is None and keypoints.shape[1] >= 3:
        scores = keypoints[:, 2]

    return keypoints[:, :2], scores


def valid_eye_pair(kpts: np.ndarray, scores: Optional[np.ndarray],
                   eye_ids: Sequence[int], kpt_thr: float) -> Optional[np.ndarray]:
    left_id, right_id = eye_ids
    if max(left_id, right_id) >= len(kpts):
        return None
    if scores is not None:
        if scores[left_id] < kpt_thr or scores[right_id] < kpt_thr:
            return None
    eyes = kpts[[left_id, right_id], :2]
    if not np.isfinite(eyes).all():
        return None
    if np.linalg.norm(eyes[0] - eyes[1]) < 1:
        return None
    return eyes


def draw_black_eye_boxes(frame: np.ndarray, instances: List[dict], args) -> int:
    boxes_drawn = 0
    height, width = frame.shape[:2]
    box_size = args.eye_box_size

    for instance in instances:
        kpts, scores = instance_keypoints_and_scores(instance)
        eyes = valid_eye_pair(kpts, scores, args.eye_ids, args.kpt_thr)
        if eyes is None:
            continue

        for eye in eyes:
            x1 = int(max(0, round(eye[0] - box_size / 2)))
            y1 = int(max(0, round(eye[1] - box_size / 2)))
            x2 = int(min(width - 1, round(eye[0] + box_size / 2)))
            y2 = int(min(height - 1, round(eye[1] + box_size / 2)))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
            boxes_drawn += 1

    return boxes_drawn


def remux_with_original_streams(input_file: str, output_rgb: str,
                                output: str, rgb_stream: str,
                                loglevel: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    spec = normalize_stream_spec(rgb_stream)
    command = [
        'ffmpeg', '-y', '-loglevel', loglevel, '-i', output_rgb,
        '-analyzeduration', '100M', '-probesize', '100M', '-i', input_file,
        '-map', '0:v:0', '-map', '1:v?', '-map', '1:a?', '-map', '1:s?',
        '-map', '1:d?', '-map', f'-1:{spec}', '-c', 'copy',
        '-allow_raw_vfw', '1', '-map_metadata', '1', '-map_chapters', '1',
        '-disposition:v', '0', '-disposition:v:0', 'default', output
    ]
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f'Final remux failed:\n{result.stderr}')


def main():
    args = parse_args()
    stream_info = ffprobe_stream(args.input, args.rgb_stream)
    width = int(stream_info['width'])
    height = int(stream_info['height'])
    fps = parse_fps(stream_info)
    frame_size = width * height * 3
    shape = (height, width, 3)

    predictions = None
    inferencer = None
    if args.predictions_json:
        predictions = load_predictions(args.predictions_json)
    else:
        from mmpose.apis.inferencers import MMPoseInferencer
        inferencer = MMPoseInferencer(
            pose2d=args.pose2d,
            pose2d_weights=args.pose2d_weights,
            det_model=args.det_model,
            det_weights=args.det_weights,
            det_cat_ids=args.det_cat_ids,
            device=args.device)

    decoder = start_decoder(args.input, args.rgb_stream, args.ffmpeg_loglevel)
    encoder = start_lossless_encoder(args.output_rgb, width, height, fps,
                                     args.ffmpeg_loglevel)

    frame_id = 0
    total_boxes = 0
    try:
        while True:
            if args.max_frames >= 0 and frame_id >= args.max_frames:
                break

            frame = read_frame(decoder, frame_size, shape)
            if frame is None:
                break

            if predictions is not None:
                instances = get_json_instances(predictions, frame_id, args)
            else:
                instances = get_pose_instances(inferencer, frame, args)
            total_boxes += draw_black_eye_boxes(frame, instances, args)
            encoder.stdin.write(frame.tobytes())
            frame_id += 1

            if frame_id % 100 == 0:
                print(f'Processed {frame_id} frames...')
    finally:
        if decoder.stdout:
            decoder.stdout.close()
        if encoder.stdin:
            encoder.stdin.close()

        decoder_return = decoder.wait()
        encoder_return = encoder.wait()

    if decoder_return != 0:
        raise RuntimeError(f'ffmpeg decoder exited with {decoder_return}.')
    if encoder_return != 0:
        raise RuntimeError(f'ffmpeg encoder exited with {encoder_return}.')

    print(f'Wrote RGB-only de-identified video: {args.output_rgb}')
    print(f'Processed {frame_id} frames; drew {total_boxes} black eye boxes.')

    if args.output:
        remux_with_original_streams(args.input, args.output_rgb, args.output,
                                    args.rgb_stream, args.ffmpeg_loglevel)
        print(f'Wrote final remuxed MKV: {args.output}')


if __name__ == '__main__':
    main()
