import math
import os
import hashlib
import shutil
import tarfile
import time
import urllib.request
import zipfile
import numpy as np
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms.functional import gaussian_blur
from modules.project_paths import DATA_DIR, resolve_path_from_code
from modules.utils import DEFAULT_RANDOM_SEED, make_torch_generator


MOZAFARI_DOG_KERNEL_SPECS = (
    (3, 3.0 / 9.0, 6.0 / 9.0),
    (3, 6.0 / 9.0, 3.0 / 9.0),
    (7, 7.0 / 9.0, 14.0 / 9.0),
    (7, 14.0 / 9.0, 7.0 / 9.0),
    (13, 13.0 / 9.0, 26.0 / 9.0),
    (13, 26.0 / 9.0, 13.0 / 9.0),
)


class DoGOnOffTransform:
    def __init__(
        self,
        sigma_center: float = 1.0,   # MNIST: 1.0 / CIFAR-10: 1.2
        center_surround_rate: float = 1.6,  # classical 1.6
        on_off_rate: float = 1.0,
        kernel_specs=None,
        normalize=True
    ):
        """
        Args:
            sigma_center: sigma for center Gaussian
            center_surround_rate: ratio to compute sigma_surround
            on_off_rate: weight for surround subtraction
        """
        self.on_off_rate = on_off_rate
        self.normalize = normalize
        if kernel_specs is None:
            self.kernel_specs = MOZAFARI_DOG_KERNEL_SPECS
        else:
            self.kernel_specs = tuple(
                (
                    int(spec['kernel_size'] if isinstance(spec, dict) else spec[0]),
                    float(spec['sigma_center'] if isinstance(spec, dict) else spec[1]),
                    float(spec['sigma_surround'] if isinstance(spec, dict) else spec[2]),
                )
                for spec in kernel_specs
            )
        if not self.kernel_specs:
            self.kernel_specs = (
                (
                    int(2 * math.ceil(3.0 * sigma_center) + 1),
                    float(sigma_center),
                    float(sigma_center * center_surround_rate),
                ),
            )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor (C,H,W) or (B,C,H,W), values in [0,1]
        Returns:
            Tensor (6C,H,W) or (B,6C,H,W) with Mozafari-style three-scale DoG.
        """

        def _minmax(z):
            zmin = z.amin(dim=(-2, -1), keepdim=True)
            zmax = z.amax(dim=(-2, -1), keepdim=True)
            epsilon = 1.0e-6
            return (z - zmin) / (zmax - zmin + epsilon)

        dog_channels = []
        for kernel_size, sigma_center, sigma_surround in self.kernel_specs:
            x_c = gaussian_blur(
                images,
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma_center, sigma_center]
            )
            x_s = gaussian_blur(
                images,
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma_surround, sigma_surround]
            )
            dog = torch.clamp(x_c - self.on_off_rate * x_s, min=0.0)
            if self.normalize:
                dog = _minmax(dog)
            dog_channels.append(dog)

        return torch.cat(dog_channels, dim=-3)


class CachedTransformDataset(Dataset):
    def __init__(self, dataset, cache_dtype=None):
        resolved_cache_dtype = _resolve_cache_dtype(cache_dtype)
        data = []
        targets = []
        dataset_len = len(dataset)
        print(f'Caching transformed dataset: {dataset_len} samples', flush=True)
        for index in range(len(dataset)):
            image, label = dataset[index]
            image = image.detach().cpu()
            if resolved_cache_dtype is not None:
                image = image.to(resolved_cache_dtype)
            data.append(image)
            targets.append(int(label))
            if (index + 1) % 10000 == 0 or index + 1 == dataset_len:
                print(
                    f'Cached transformed dataset: {index + 1}/{dataset_len}',
                    flush=True,
                )

        self.data = torch.stack(data)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index])


class AugmentedDataset(Dataset):
    """Apply a stochastic tensor transform without changing the base dataset."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label


def _file_md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


DOWNLOAD_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; BoyoNet/1.0)',
}


def _archive_is_current(path, md5):
    return path.is_file() and (md5 is None or _file_md5(path) == md5)


class DownloadLock:
    def __init__(self, path, stale_seconds=6 * 60 * 60):
        self.path = Path(str(path) + '.lock')
        self.stale_seconds = stale_seconds
        self.fd = None

    def __enter__(self):
        last_log = 0.0
        while True:
            try:
                self.fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                )
                os.write(self.fd, f'{os.getpid()}\n'.encode('ascii'))
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                now = time.monotonic()
                if now - last_log >= 60:
                    print(f'Waiting for download lock: {self.path}', flush=True)
                    last_log = now
                time.sleep(5)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


def _download_file(url, path, md5=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if _archive_is_current(path, md5):
        return

    with DownloadLock(path):
        if _archive_is_current(path, md5):
            return

        tmp_path = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        tmp_path.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers=DOWNLOAD_HEADERS)
        print(f'Downloading {url} -> {path}', flush=True)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                print(
                    f'Download response: {response.status} {response.geturl()}',
                    flush=True,
                )
                with tmp_path.open('wb') as file:
                    shutil.copyfileobj(response, file)

            if tmp_path.stat().st_size == 0:
                raise RuntimeError(f'Downloaded empty file for {path.name}: {url}')

            if md5 is not None:
                actual_md5 = _file_md5(tmp_path)
                if actual_md5 != md5:
                    raise RuntimeError(
                        f'MD5 mismatch for {path.name}: expected {md5}, got {actual_md5}'
                    )
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def _assert_extract_path_safe(destination, member_name):
    destination = destination.resolve()
    member_path = (destination / member_name).resolve()
    if destination != member_path and destination not in member_path.parents:
        raise RuntimeError(f'Unsafe archive member path: {member_name}')


def _extract_zip(zip_path, destination):
    print(f'Extracting {zip_path}', flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            _assert_extract_path_safe(destination, member)
        archive.extractall(destination)


def _extract_tar(tar_path, destination):
    print(f'Extracting {tar_path}', flush=True)
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            _assert_extract_path_safe(destination, member.name)
        archive.extractall(destination)


def _folder_contains_at_least(folder, suffix, min_count):
    if not folder.is_dir():
        return False
    count = 0
    for path in folder.rglob(f'*{suffix}'):
        if path.is_file():
            count += 1
            if count >= min_count:
                return True
    return False


def _events_to_count_frame(
    x, y, p, height, width, spatial_downsample=1, t=None, temporal_bins=1,
    representation='normalized_count'
):
    spatial_downsample = int(spatial_downsample)
    if spatial_downsample <= 0:
        raise ValueError('spatial_downsample must be positive')
    temporal_bins = int(temporal_bins)
    if temporal_bins <= 0:
        raise ValueError('temporal_bins must be positive')
    representation = str(representation).strip().lower()
    if representation not in {'normalized_count', 'binary'}:
        raise ValueError(
            "representation must be 'normalized_count' or 'binary'"
        )

    out_height = int(math.ceil(height / spatial_downsample))
    out_width = int(math.ceil(width / spatial_downsample))
    x = np.asarray(x, dtype=np.int64) // spatial_downsample
    y = np.asarray(y, dtype=np.int64) // spatial_downsample
    p = np.asarray(p, dtype=np.int64)
    if temporal_bins > 1:
        if t is None:
            raise ValueError('temporal_bins > 1 requires event timestamps')
        t = np.asarray(t, dtype=np.float64)
    valid = (
        (x >= 0) & (x < out_width)
        & (y >= 0) & (y < out_height)
        & (p >= 0) & (p < 2)
    )

    frame = np.zeros((2 * temporal_bins, out_height, out_width), dtype=np.float32)
    if np.any(valid):
        if temporal_bins == 1:
            channel = p[valid]
        else:
            valid_t = t[valid]
            t_min = float(valid_t.min())
            t_max = float(valid_t.max())
            if t_max > t_min:
                temporal_index = np.floor(
                    (valid_t - t_min) / (t_max - t_min) * temporal_bins
                ).astype(np.int64)
                temporal_index = np.clip(temporal_index, 0, temporal_bins - 1)
            else:
                temporal_index = np.zeros_like(valid_t, dtype=np.int64)
            channel = p[valid] + 2 * temporal_index
        np.add.at(frame, (channel, y[valid], x[valid]), 1.0)
        if representation == 'binary':
            frame = (frame > 0.0).astype(np.float32)
        else:
            max_count = float(frame.max())
            if max_count > 0.0:
                frame /= max_count
    return torch.from_numpy(frame)


class NMNISTEventFrameDataset(Dataset):
    train_url = (
        'https://data.mendeley.com/public-files/datasets/468j46mzdv/files/'
        '39c25547-014b-4137-a934-9d29fa53c7a0/file_downloaded'
    )
    test_url = (
        'https://data.mendeley.com/public-files/datasets/468j46mzdv/files/'
        '05a4d654-7e03-4c15-bdfa-9bb2bcbea494/file_downloaded'
    )
    train_filename = 'train.zip'
    test_filename = 'test.zip'
    train_md5 = '20959b8e626244a1b502305a9e6e2031'
    test_md5 = '69ca8762b2fe404d9b9bad1103e97832'
    sensor_size = (34, 34)

    def __init__(
        self,
        root,
        train=True,
        download=True,
        spatial_downsample=1,
        first_saccade_only=False,
        temporal_bins=1,
        representation='normalized_count',
    ):
        self.root = Path(root) / 'NMNIST'
        self.train = bool(train)
        self.spatial_downsample = int(spatial_downsample)
        self.first_saccade_only = bool(first_saccade_only)
        self.temporal_bins = int(temporal_bins)
        self.representation = str(representation)
        self.folder_name = 'Train' if self.train else 'Test'
        self.filename = self.train_filename if self.train else self.test_filename
        self.url = self.train_url if self.train else self.test_url
        self.md5 = self.train_md5 if self.train else self.test_md5
        self.folder = self.root / self.folder_name

        if not _folder_contains_at_least(self.folder, '.bin', 10000):
            if not download:
                raise FileNotFoundError(f'N-MNIST folder not found: {self.folder}')
            archive_path = self.root / self.filename
            _download_file(self.url, archive_path, md5=self.md5)
            _extract_zip(archive_path, self.root)

        self.data = []
        self.targets = []
        for path in sorted(self.folder.rglob('*.bin')):
            label = int(path.parent.name)
            self.data.append(path)
            self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def _read_bin_events(self, path):
        raw = np.fromfile(path, dtype=np.uint8).astype(np.uint32)
        if raw.size % 5 != 0:
            raise RuntimeError(f'Invalid N-MNIST event file length: {path}')
        raw = raw.reshape(-1, 5)
        x = raw[:, 0]
        y = raw[:, 1]
        p = (raw[:, 2] & 128) >> 7
        timestamps = (
            ((raw[:, 2] & 127) << 16) | (raw[:, 3] << 8) | raw[:, 4]
        ).astype(np.int64)
        for overflow_index in np.where(y == 240)[0]:
            timestamps[overflow_index:] += 2**13
        valid = y != 240
        if self.first_saccade_only:
            valid = valid & (timestamps < 100000)
        return x[valid], y[valid], p[valid], timestamps[valid]

    def __getitem__(self, index):
        x, y, p, t = self._read_bin_events(self.data[index])
        height, width = self.sensor_size
        image = _events_to_count_frame(
            x,
            y,
            p,
            height=height,
            width=width,
            spatial_downsample=self.spatial_downsample,
            t=t,
            temporal_bins=self.temporal_bins,
            representation=self.representation,
        )
        return image, int(self.targets[index])


class DVS128GestureEventFrameDataset(Dataset):
    train_url = 'https://ndownloader.figshare.com/files/38022171'
    test_url = 'https://ndownloader.figshare.com/files/38020584'
    train_filename = 'ibmGestureTrain.tar.gz'
    test_filename = 'ibmGestureTest.tar.gz'
    train_md5 = '3a8f0d4120a166bac7591f77409cb105'
    test_md5 = '56070e45dadaa85fff82e0fbfbc06de5'
    sensor_size = (128, 128)

    def __init__(
        self, root, train=True, download=True, spatial_downsample=4,
        temporal_bins=1, representation='normalized_count'
    ):
        self.root = Path(root) / 'DVS128Gesture'
        self.train = bool(train)
        self.spatial_downsample = int(spatial_downsample)
        self.temporal_bins = int(temporal_bins)
        self.representation = str(representation)
        self.folder_name = 'ibmGestureTrain' if self.train else 'ibmGestureTest'
        self.filename = self.train_filename if self.train else self.test_filename
        self.url = self.train_url if self.train else self.test_url
        self.md5 = self.train_md5 if self.train else self.test_md5
        self.folder = self.root / self.folder_name

        if not _folder_contains_at_least(self.folder, '.npy', 100):
            if not download:
                raise FileNotFoundError(
                    f'DVS128 Gesture folder not found: {self.folder}'
                )
            archive_path = self.root / self.filename
            _download_file(self.url, archive_path, md5=self.md5)
            _extract_tar(archive_path, self.root)

        self.data = []
        self.targets = []
        for path in sorted(self.folder.rglob('*.npy')):
            self.data.append(path)
            self.targets.append(int(path.stem))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        events = np.load(self.data[index])
        timestamps = events[:, 3] if events.shape[1] > 3 else None
        height, width = self.sensor_size
        image = _events_to_count_frame(
            events[:, 0],
            events[:, 1],
            events[:, 2],
            height=height,
            width=width,
            spatial_downsample=self.spatial_downsample,
            t=timestamps,
            temporal_bins=self.temporal_bins,
            representation=self.representation,
        )
        return image, int(self.targets[index])


EVENT_FRAME_DATASETS = {
    'NMNIST': NMNISTEventFrameDataset,
    'N-MNIST': NMNISTEventFrameDataset,
    'DVS128Gesture': DVS128GestureEventFrameDataset,
    'DVS128_Gesture': DVS128GestureEventFrameDataset,
}


def _resolve_cache_dtype(cache_dtype):
    if cache_dtype is None:
        return None
    if isinstance(cache_dtype, torch.dtype):
        return cache_dtype
    dtype_name = str(cache_dtype).strip().lower()
    dtype_map = {
        'float32': torch.float32,
        'float': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'half': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            'cache_dtype must be one of float32, float16, bfloat16, or null'
        )
    return dtype_map[dtype_name]


def _resolve_num_workers(num_workers):
    if num_workers is not None:
        return int(num_workers)

    env_value = os.environ.get('BOYONET_DATALOADER_NUM_WORKERS')
    if env_value is None or str(env_value).strip() == '':
        return min(12, os.cpu_count() or 0)

    try:
        resolved_num_workers = int(env_value)
    except ValueError as exc:
        raise ValueError(
            'BOYONET_DATALOADER_NUM_WORKERS must be a non-negative integer'
        ) from exc
    if resolved_num_workers < 0:
        raise ValueError(
            'BOYONET_DATALOADER_NUM_WORKERS must be a non-negative integer'
        )
    return resolved_num_workers


def _get_augmentation_transforms(augmentation_settings):
    if not augmentation_settings:
        return []

    settings = dict(augmentation_settings)
    if not settings.get('enabled', True):
        return []

    transforms = []
    random_crop = settings.get('random_crop')
    if random_crop:
        if isinstance(random_crop, dict):
            size = int(random_crop.get('size', 32))
            padding = int(random_crop.get('padding', 4))
        else:
            size = int(settings.get('random_crop_size', 32))
            padding = int(settings.get('random_crop_padding', 4))
        transforms.append(torchvision.transforms.RandomCrop(size, padding=padding))

    random_horizontal_flip = settings.get('random_horizontal_flip')
    if random_horizontal_flip:
        if isinstance(random_horizontal_flip, dict):
            probability = float(random_horizontal_flip.get('p', 0.5))
        else:
            probability = float(settings.get('random_horizontal_flip_p', 0.5))
        transforms.append(torchvision.transforms.RandomHorizontalFlip(p=probability))

    return transforms


class RepeatInputChannels:
    """Duplicate input planes without introducing new image information."""
    def __init__(self, repeats):
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError('input_channel_repeat must be a positive integer')
        self.repeats = repeats

    def __call__(self, image):
        return image if self.repeats == 1 else image.repeat(self.repeats, 1, 1)


def _get_image_transform(use_DoG, DoG_settings, pre_tensor_transforms=None,
                         input_channel_repeat=1):
    transforms = list(pre_tensor_transforms or [])
    transforms.append(torchvision.transforms.ToTensor())
    if use_DoG:
        transforms.append(DoGOnOffTransform(**DoG_settings))
    transforms.append(RepeatInputChannels(input_channel_repeat))
    return torchvision.transforms.Compose(transforms)


def get_processed_dataloaders(
    dataset_name, batch_size, use_DoG, DoG_settings, label_perm=False,
    num_workers=None, pin_memory=None, data_root=None, valid_train_ratio=0.1,
    seed=DEFAULT_RANDOM_SEED, cache_transforms=False, cache_dtype=None,
    augmentation_settings=None, event_frame_settings=None, input_channel_repeat=1
):
    num_workers = _resolve_num_workers(num_workers)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    data_root = DATA_DIR if data_root is None else resolve_path_from_code(data_root)
    print(
        f'Dataloader settings: num_workers={num_workers}, pin_memory={pin_memory}, '
        f'cache_transforms={cache_transforms}, cache_dtype={cache_dtype}',
        flush=True,
    )

    if dataset_name in EVENT_FRAME_DATASETS:
        if input_channel_repeat != 1:
            raise ValueError('input_channel_repeat is only supported for image datasets')
        return _get_event_frame_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            use_DoG=use_DoG,
            label_perm=label_perm,
            num_workers=num_workers,
            pin_memory=pin_memory,
            data_root=data_root,
            valid_train_ratio=valid_train_ratio,
            seed=seed,
            cache_transforms=cache_transforms,
            cache_dtype=cache_dtype,
            augmentation_settings=augmentation_settings,
            event_frame_settings=event_frame_settings,
        )

    dataset_class = getattr(torchvision.datasets, dataset_name)
    train_pre_tensor_transforms = _get_augmentation_transforms(
        augmentation_settings
    )
    train_transform = _get_image_transform(
        use_DoG,
        DoG_settings,
        pre_tensor_transforms=train_pre_tensor_transforms,
        input_channel_repeat=input_channel_repeat
    )
    eval_transform = _get_image_transform(
        use_DoG, DoG_settings, input_channel_repeat=input_channel_repeat
    )
    has_train_only_augmentation = len(train_pre_tensor_transforms) > 0

    train_dataset = dataset_class(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=True
    )
    valid_dataset = train_dataset
    if has_train_only_augmentation:
        valid_dataset = dataset_class(
            root=str(data_root),
            train=True,
            transform=eval_transform,
            download=True
        )
    test_dataset = dataset_class(
        root=str(data_root),
        train=False,
        transform=eval_transform,
        download=True
    )

    if label_perm:
        # Generate label mapping (shuffle 0–9)
        num_classes = len(set(train_dataset.targets.numpy()))
        perm = torch.randperm(num_classes, generator=make_torch_generator(seed))

        # Remap labels using the shuffled label mapping
        train_dataset.targets = perm[train_dataset.targets]
        test_dataset.targets = perm[test_dataset.targets]

    if cache_transforms:
        resolved_cache_dtype = _resolve_cache_dtype(cache_dtype)
        train_dataset = CachedTransformDataset(
            train_dataset,
            cache_dtype=resolved_cache_dtype
        )
        if has_train_only_augmentation:
            valid_dataset = CachedTransformDataset(
                valid_dataset,
                cache_dtype=resolved_cache_dtype
            )
        else:
            valid_dataset = train_dataset
        test_dataset = CachedTransformDataset(
            test_dataset,
            cache_dtype=resolved_cache_dtype
        )

    train_size = int(len(train_dataset) * (1-valid_train_ratio))
    valid_size = int(len(train_dataset) * valid_train_ratio)
    if has_train_only_augmentation:
        indices = torch.randperm(
            len(train_dataset),
            generator=make_torch_generator(seed)
        ).tolist()
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        train_dataset = Subset(train_dataset, train_indices)
        valid_dataset = Subset(valid_dataset, valid_indices)
    else:
        train_dataset, valid_dataset = random_split(
            train_dataset,
            [train_size, valid_size],
            generator=make_torch_generator(seed)
        )

    datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }

    dataloaders = {}
    for phase_idx, (phase, dataset) in enumerate(datasets.items()):
        dataloaders[phase] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=make_torch_generator(seed + phase_idx)
        )

    return dataloaders


def _apply_label_permutation(train_dataset, test_dataset, seed):
    train_targets = torch.as_tensor(train_dataset.targets, dtype=torch.long)
    test_targets = torch.as_tensor(test_dataset.targets, dtype=torch.long)
    num_classes = int(train_targets.max().item()) + 1
    perm = torch.randperm(num_classes, generator=make_torch_generator(seed))
    train_dataset.targets = perm[train_targets].tolist()
    test_dataset.targets = perm[test_targets].tolist()


def _get_event_frame_dataloaders(
    dataset_name, batch_size, use_DoG, label_perm, num_workers, pin_memory,
    data_root, valid_train_ratio, seed, cache_transforms, cache_dtype,
    augmentation_settings=None, event_frame_settings=None
):
    if use_DoG:
        raise ValueError('Event-frame datasets should use use_DoG: false')

    dataset_class = EVENT_FRAME_DATASETS[dataset_name]
    dataset_settings = dict(event_frame_settings or {})
    train_dataset = dataset_class(
        root=data_root,
        train=True,
        download=True,
        **dataset_settings,
    )
    test_dataset = dataset_class(
        root=data_root,
        train=False,
        download=True,
        **dataset_settings,
    )
    if label_perm:
        _apply_label_permutation(train_dataset, test_dataset, seed)

    if cache_transforms:
        resolved_cache_dtype = _resolve_cache_dtype(cache_dtype)
        train_dataset = CachedTransformDataset(
            train_dataset,
            cache_dtype=resolved_cache_dtype
        )
        test_dataset = CachedTransformDataset(
            test_dataset,
            cache_dtype=resolved_cache_dtype
        )

    train_size = int(len(train_dataset) * (1 - valid_train_ratio))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(
        train_dataset,
        [train_size, valid_size],
        generator=make_torch_generator(seed)
    )

    train_transforms = _get_augmentation_transforms(augmentation_settings)
    if train_transforms:
        train_dataset = AugmentedDataset(
            train_dataset,
            torchvision.transforms.Compose(train_transforms),
        )

    datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset,
    }
    dataloaders = {}
    for phase_idx, (phase, dataset) in enumerate(datasets.items()):
        dataloaders[phase] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=make_torch_generator(seed + phase_idx),
        )
    return dataloaders
