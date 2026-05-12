# Copyright (c) OpenMMLab. All rights reserved.
"""Batch de-identify MKV files using matching per-video pose JSON files.

Example:
    PYTHONPATH=. python demo/deidentify_rgb_batch.py \
        /path/to/mkvs \
        /path/to/jsons \
        /path/to/deidentified_mkvs \
        --rgb-stream v:0 \
        --eye-box-size 8
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing input .mkv files.')
    parser.add_argument(
        'predictions_dir',
        help='Directory containing matching .json pose prediction files.')
    parser.add_argument(
        'output_dir',
        help='Directory where RGB-only de-identified .mkv files will be '
        'written.')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search input_dir for .mkv files. Relative subdirectory '
        'layout is preserved for JSON lookup and outputs.')
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue processing remaining videos if one video fails.')
    parser.add_argument(
        '--rgb-stream',
        default='v:0',
        help='FFmpeg stream specifier for the RGB stream. Defaults to v:0.')
    parser.add_argument(
        '--allow-missing-predictions',
        action='store_true',
        default=True,
        help='Allow frames missing from matching JSON files to pass through '
        'without redaction. Enabled by default.')
    parser.add_argument(
        '--no-allow-missing-predictions',
        action='store_false',
        dest='allow_missing_predictions',
        help='Raise an error if any frame is missing from the matching JSON '
        'file.')
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


def find_input_videos(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = '**/*.mkv' if recursive else '*.mkv'
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def build_command(script_path: Path, input_video: Path, predictions_json: Path,
                  output_video: Path, args) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        str(input_video),
        '--predictions-json',
        str(predictions_json),
        '--output-rgb',
        str(output_video),
        '--rgb-stream',
        args.rgb_stream,
        '--eye-ids',
        str(args.eye_ids[0]),
        str(args.eye_ids[1]),
        '--kpt-thr',
        str(args.kpt_thr),
        '--eye-box-size',
        str(args.eye_box_size),
        '--max-frames',
        str(args.max_frames),
        '--ffmpeg-loglevel',
        args.ffmpeg_loglevel,
    ]
    if args.allow_missing_predictions:
        command.append('--allow-missing-predictions')
    return command


def run_command(command: Sequence[str]) -> int:
    process = subprocess.run(command)
    return process.returncode


def validate_inputs(input_dir: Path, predictions_dir: Path,
                    output_dir: Path) -> None:
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {input_dir}')
    if not predictions_dir.is_dir():
        raise NotADirectoryError(
            f'Predictions directory does not exist: {predictions_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    validate_inputs(input_dir, predictions_dir, output_dir)

    videos = find_input_videos(input_dir, args.recursive)
    if not videos:
        raise RuntimeError(f'No .mkv files found in {input_dir}.')

    missing_json = []
    jobs = []
    for input_video in videos:
        relative_video = input_video.relative_to(input_dir)
        relative_json = relative_video.with_suffix('.json')
        predictions_json = predictions_dir / relative_json
        output_video = output_dir / relative_video

        if not predictions_json.is_file():
            missing_json.append((input_video, predictions_json))
            continue
        if output_video.exists():
            print(f'Skipping existing output: {output_video}')
            continue
        jobs.append((input_video, predictions_json, output_video))

    for input_video, predictions_json in missing_json:
        print(f'Skipping missing JSON: {input_video} -> {predictions_json}')

    if not jobs:
        print('No videos to process.')
        return

    script_path = Path(__file__).with_name('deidentify_rgb.py')
    failures = []
    for index, (input_video, predictions_json,
                output_video) in enumerate(jobs, start=1):
        output_video.parent.mkdir(parents=True, exist_ok=True)

        print(f'[{index}/{len(jobs)}] De-identifying {input_video}')
        command = build_command(script_path, input_video, predictions_json,
                                output_video, args)
        returncode = run_command(command)
        if returncode != 0:
            failures.append((input_video, returncode))
            if not args.continue_on_error:
                break

    if failures:
        message = '\n'.join(
            f'  {video} exited with status {returncode}'
            for video, returncode in failures)
        raise RuntimeError(
            f'Failed to de-identify {len(failures)} video(s):\n{message}')

    print(f'De-identified {len(jobs)} video(s) into {output_dir}')


if __name__ == '__main__':
    main()
