"""Create a registered dataset from a folder of CT scans.

Usage:
    python registered_dataset.py --data-root <folder> [--reference-volume <scan-folder>]
"""

import argparse
import json
import os
import random
import shutil
from glob import glob

import nibabel as nib
import numpy as np
from monai import metrics

from ImageRegistration import ImageRegistrator
from utils import array_to_tensor, find_shifts, flip_volume, read_dicom, rolled_ssim
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Default parameters for file selection
data_dir = None
reference_volume_dir = None
scanners_list = ['A1', 'A2', 'B1', 'B2', 'C1', 'D1', 'E1', 'E2', 'F1', 'G1', 'G2', 'H1', 'H2']
thickness = [2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.0, 2.5, 2.0, 2.0, 2.0, 2.0, 2.0]
slice_thinknesses = {scanners_list[i]: thickness[i] for i in range(len(scanners_list))}
scanners = '*'
dose = '*'
reconstruction_method = '*'
dataset_dir = os.path.join(os.path.dirname(__file__), 'registered_dataset_output')
registration_mode = 'identity'
ssim_data_range = 2000
#crop_region = [20,330,120,395,64,445]
crop_region = [13,323,120,395,64,445]
downsample_factor = 1


def _has_dicom_files(scan_folder):
    return any(file_name.lower().endswith('.dcm') for file_name in os.listdir(scan_folder))


def _find_from_studies_map(data_root):
    map_path = os.path.join(data_root, 'studies_map.json')
    if not os.path.exists(map_path):
        return []

    with open(map_path, 'r', encoding='utf-8') as handle:
        studies_map = json.load(handle)

    scan_folders = []
    for study_id, entry in studies_map.items():
        image_path = entry.get('image')
        if not image_path:
            continue

        candidate_paths = []
        if os.path.isabs(image_path):
            candidate_paths.append(image_path)
            candidate_paths.append(os.path.join(data_root, study_id, os.path.basename(image_path)))
            candidate_paths.append(os.path.join(data_root, study_id, os.path.basename(os.path.dirname(image_path))))
        else:
            candidate_paths.append(os.path.join(data_root, image_path))
            candidate_paths.append(os.path.join(data_root, study_id, os.path.basename(image_path)))

        for candidate_path in candidate_paths:
            if os.path.isdir(candidate_path) and _has_dicom_files(candidate_path):
                scan_folders.append(candidate_path)
                break

    return sorted(set(scan_folders))


def _find_scan_folders(data_root):
    scan_folders = _find_from_studies_map(data_root)
    if scan_folders:
        return scan_folders

    fallback_scan_folders = []
    for current_root, _, files in os.walk(data_root):
        if any(file_name.lower().endswith('.dcm') for file_name in files):
            lower_root = current_root.lower()
            if 'mask' in lower_root or 'seg' in lower_root or '.nii' in lower_root:
                continue
            fallback_scan_folders.append(current_root)

    return sorted(set(fallback_scan_folders))


def _scan_output_name(scan_folder, data_root):
    relative_path = os.path.relpath(scan_folder, data_root)
    return relative_path.replace(os.sep, '__') + '.nii.gz'


def _process_single_scan(scan_folder, folder, reference_volumes, output_dir, max_dicom_files, smoke_test, registration_mode, downsample_factor, ssim_data_range):
    try:
        print(f'Processing (worker) {scan_folder} ...')
        final_file = os.path.join(output_dir, _scan_output_name(scan_folder, folder))
        done_marker = final_file + '.done'
        # If either the final file or a done marker exists, treat as already completed
        if os.path.exists(final_file) or os.path.exists(done_marker):
            return ('skipped', scan_folder, None)

        volumes = read_dicom(scan_folder, numpy_format=True, crop_region=None, slice_thinknesses=slice_thinknesses, max_files=max_dicom_files, quick=smoke_test)
        nifti_image = volumes[-1]

        if smoke_test:
            registered_nifti = nib.Nifti1Image(volumes[1].astype(float), nifti_image.affine)
            temp_file = final_file.replace(".nii.gz", ".partial.nii.gz")
            # write to a temporary file on the same filesystem, then atomically replace
            registered_nifti.to_filename(temp_file)
            try:
                os.replace(temp_file, final_file)
                # create done marker
                with open(done_marker, 'w', encoding='utf-8'):
                    pass
            finally:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
            return ('ok', scan_folder, None)

        ssim = metrics.SSIMMetric(spatial_dims=3, data_range=ssim_data_range)
        _ssim = ssim(reference_volumes[0], volumes[0]).item()

        _ssim0 = rolled_ssim(reference_volumes[0], volumes[0]).item()
        _ssim1 = rolled_ssim(reference_volumes[0], volumes[2]).item()
        if _ssim1 > _ssim0:
            volumes[0] = volumes[2]
            volumes[1] = volumes[3]

        volumes[0], shift, _ = find_shifts(volumes[0], reference_volumes[0], axis=-1)
        volumes[1] = np.roll(volumes[1], shift, axis=0)

        volume = volumes[1][crop_region[0]:crop_region[1], crop_region[2]:crop_region[3], crop_region[4]:crop_region[5]]
        reference_volume_cropped = reference_volumes[1][crop_region[0]:crop_region[1], crop_region[2]:crop_region[3], crop_region[4]:crop_region[5]]

        registrator = ImageRegistrator(registration_mode, reference_volumes[1])
        registered_image = registrator.register_image(volume, reference_volume_cropped, downsample_factor=downsample_factor)[0]

        registered_image_tensor = array_to_tensor(registered_image.transpose(1, 2, 0))
        reference_volume_tensor = array_to_tensor(reference_volume_cropped.transpose(1, 2, 0))

        _ssim_after = ssim(reference_volume_tensor, registered_image_tensor).item()
        if _ssim_after < _ssim:
            registered_image = volume

        registered_image_nifti = registered_image.transpose(2, 1, 0)
        registered_image_nifti = flip_volume(registered_image_nifti, axis=1)

        registered_nifti = nib.Nifti1Image(registered_image_nifti.astype(float), nifti_image.affine)
        temp_file = final_file.replace(".nii.gz", ".partial.nii.gz")
        registered_nifti.to_filename(temp_file)
        try:
            os.replace(temp_file, final_file)
            # write done marker for robust detection
            with open(done_marker, 'w', encoding='utf-8'):
                pass
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
        return ('ok', scan_folder, None)
    except Exception as exc:
        # attempt to remove any partial file left behind
        try:
            final_file = os.path.join(output_dir, _scan_output_name(scan_folder, folder))
            temp_file = final_file.replace(".nii.gz", ".partial.nii.gz")
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass
        return ('failed', scan_folder, str(exc))


def _print_progress(completed, total):
    pct = (completed / total) * 100 if total else 100.0
    bar_len = 30
    filled = int(bar_len * completed / total) if total else bar_len
    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
    print(f'Progress: {completed}/{total} {bar} {pct:5.1f}%')


def create_registered_dataset(folder, reference_volume=None, output_dir=None, limit=None, max_dicom_files=None, smoke_test=False, workers=1, registration_mode_arg=None, downsample_factor=1):

    # Define the SSIM
    ssim = metrics.SSIMMetric(spatial_dims=3, data_range=ssim_data_range)

    scan_folders = _find_scan_folders(folder)
    if not scan_folders:
        raise ValueError(f'No scan folders were found under {folder}')

    if reference_volume is None:
        reference_volume = scan_folders[0]
        print(f'No reference volume provided. Using {reference_volume} as the reference.')

    # Read the reference volume
    reference_volumes = read_dicom(reference_volume, numpy_format=True, crop_region=None, slice_thinknesses=slice_thinknesses, max_files=max_dicom_files, quick=smoke_test)

    # Randomly shuffle the list
    random.shuffle(scan_folders)
    if limit is not None:
        scan_folders = scan_folders[:limit]

    # Create the registered dataset directory:
    output_dir = output_dir or dataset_dir
    os.makedirs(output_dir, exist_ok=True)

    failed_scans = []
    skipped_scans = []

    # Print the ground truth directory
    print(f'Reference Volume: {reference_volume}')
    print(f'Output Directory: {output_dir}')
    print(f'Found {len(scan_folders)} scan folders to process.')
    if smoke_test:
        print('Smoke test mode enabled. Skipping registration and similarity scoring.')

    # Determine registration mode to use
    registration_mode_used = registration_mode_arg or registration_mode

    # Use workers if requested
    total_scans = len(scan_folders)
    completed = 0
    if workers is None or int(workers) <= 1:
        # fallback to serial processing
        for scan_folder in scan_folders:
            status, folder_processed, err = _process_single_scan(scan_folder, folder, reference_volumes, output_dir, max_dicom_files, smoke_test, registration_mode_used, downsample_factor, ssim_data_range)
            completed += 1
            if status == 'skipped':
                skipped_scans.append(folder_processed)
            elif status == 'failed':
                failed_scans.append((folder_processed, err))
            _print_progress(completed, total_scans)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = {}
        worker_fn = partial(_process_single_scan, folder=folder, reference_volumes=reference_volumes, output_dir=output_dir, max_dicom_files=max_dicom_files, smoke_test=smoke_test, registration_mode=registration_mode_used, downsample_factor=downsample_factor, ssim_data_range=ssim_data_range)
        for scan_folder in scan_folders:
            futures[pool.submit(worker_fn, scan_folder)] = scan_folder

        for fut in as_completed(futures):
            status, folder_processed, err = fut.result()
            completed += 1
            if status == 'skipped':
                skipped_scans.append(folder_processed)
            elif status == 'failed':
                failed_scans.append((folder_processed, err))
            _print_progress(completed, total_scans)

    summary_file = os.path.join(output_dir, 'failed_scans.txt')
    with open(summary_file, 'w', encoding='utf-8') as handle:
        handle.write(f'Total scans found: {len(scan_folders)}\n')
        handle.write(f'Skipped scans: {len(skipped_scans)}\n')
        handle.write(f'Failed scans: {len(failed_scans)}\n\n')
        for scan_folder, error_message in failed_scans:
            handle.write(f'{scan_folder}\t{error_message}\n')

    print(f'Finished. Total scans: {len(scan_folders)}')
    print(f'Skipped scans: {len(skipped_scans)}')
    print(f'Failed scans: {len(failed_scans)}')
    if failed_scans:
        print(f'Failure list saved to {summary_file}')


def parse_args():
    parser = argparse.ArgumentParser(description='Create a registered CT dataset from DICOM scan folders.')
    parser.add_argument('--data-root', required=True, help='Folder that contains the raw scan folders.')
    parser.add_argument('--reference-volume', default=None, help='Folder used as the reference scan.')
    parser.add_argument('--output-dir', default=dataset_dir, help='Directory where registered NIfTI files are written.')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on how many scans to process.')
    parser.add_argument('--max-dicom-files', type=int, default=None, help='Optional cap on DICOM files read per scan for a faster smoke test.')
    parser.add_argument('--smoke-test', action='store_true', help='Skip registration and scoring so the test finishes quickly.')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes to use for parallel processing.')
    parser.add_argument('--registration-mode', choices=['identity', 'elastic', 'ants'], default=None, help='Registration backend to use.')
    parser.add_argument('--downsample-factor', type=int, default=1, help='Downsample factor to speed registration.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_registered_dataset(args.data_root, reference_volume=args.reference_volume, output_dir=args.output_dir, limit=args.limit, max_dicom_files=args.max_dicom_files, smoke_test=args.smoke_test, workers=args.workers, registration_mode_arg=args.registration_mode, downsample_factor=args.downsample_factor)