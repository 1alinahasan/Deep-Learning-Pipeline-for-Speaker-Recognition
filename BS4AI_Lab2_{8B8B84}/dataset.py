import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchaudio.compliance.kaldi import fbank as kaldi_fbank


# Question 1
def load_voxceleb(metadata_file, data_directory):
    metadata = pd.read_csv(metadata_file, engine="python")
    data_directory = os.path.abspath(data_directory)

    video_index = {}

    # Walk through speaker/video folders and index flac files
    for root, _, files in os.walk(data_directory):
        flacs = [f for f in files if f.endswith(".flac")]
        if not flacs:
            continue

        parts = os.path.normpath(root).split(os.sep)
        if len(parts) < 2:
            continue

        speaker_id = parts[-2]
        video_id = parts[-1]

        if speaker_id.startswith("id"):
            video_index[(speaker_id, video_id)] = {
                "dir": root,
                "files": set(flacs),
            }

    rows = []
    for _, row in metadata.iterrows():
        speaker_id = str(row["speaker"])
        video_id = str(row["video_id"])
        file_id = str(row["id"])
        utt = file_id.split("_")[-1]

        audio_path = ""
        info = video_index.get((speaker_id, video_id))

        if info is not None:
            vdir = info["dir"]
            files = info["files"]

            if f"{file_id}.flac" in files:
                audio_path = os.path.join(vdir, f"{file_id}.flac")
            elif f"{utt}.flac" in files:
                audio_path = os.path.join(vdir, f"{utt}.flac")

        # Skip rows without valid audio
        if audio_path == "" or not os.path.isfile(audio_path):
            continue

        rows.append({
            "id": file_id,
            "speaker_id": speaker_id,
            "video_id": video_id,
            "sampling_rate": row["sample_freq"],
            "duration": row["duration"],
            "audio_path": audio_path,
            "gender": row["gender"],
            "age": row["age"],
        })

    return pd.DataFrame(rows)


# Question 2 & 5
class My_Dataset(Dataset):
    def __init__(self, dataframe, min_len=2.0, max_len=5.0, sample_rate=16000):
        self.df = dataframe.reset_index(drop=True)
        self.sample_rate = sample_rate

        self.min_samples = int(min_len * sample_rate)
        self.max_samples = int(max_len * sample_rate)

        speakers = sorted(self.df["speaker_id"].unique())
        self.speaker_to_idx = {s: i for i, s in enumerate(speakers)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load audio
        try:
            waveform, sr = torchaudio.load(row["audio_path"])
        except Exception:
            waveform = torch.zeros(1, self.max_samples)
            sr = self.sample_rate

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        T = waveform.shape[1]

        # Enforce minimum length
        if T < self.min_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.min_samples - T)
            )
            T = waveform.shape[1]

        # Crop or pad to target length
        if T > self.max_samples:
            start = torch.randint(0, T - self.max_samples + 1, (1,)).item()
            waveform = waveform[:, start:start + self.max_samples]
        elif T < self.max_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.max_samples - T)
            )

        speaker = torch.tensor(
            self.speaker_to_idx[row["speaker_id"]],
            dtype=torch.long
        )

        age = float(row["age"]) if pd.notna(row["age"]) else -1.0
        age = torch.tensor(age, dtype=torch.float)

        gender = str(row["gender"]).lower().startswith("m")
        gender = torch.tensor(1 if gender else 0, dtype=torch.long)

        return waveform.float(), speaker, age, gender

# Question 3
def train_test_split(dataframe, per_speaker=True, prop=0.9):
    if not per_speaker:
        shuffled = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(len(shuffled) * prop)
        return shuffled[:split], shuffled[split:]

    speakers = dataframe["speaker_id"].unique()
    np.random.seed(42)
    np.random.shuffle(speakers)

    split = int(len(speakers) * prop)
    train_speakers = set(speakers[:split])

    train_df = dataframe[dataframe["speaker_id"].isin(train_speakers)]
    test_df = dataframe[~dataframe["speaker_id"].isin(train_speakers)]

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# Question 7 (Lab 2 Q1 )
class My_Collator:
    def __init__(self, sample_rate=16000, num_mel_bins=24, frame_length_ms=25.0, frame_shift_ms=10.0, n_mels=None):
        if n_mels is not None:
            num_mel_bins = n_mels
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms

    def __call__(self, batch):
        waveforms, speakers, ages, genders = zip(*batch)

        fbanks = []
        lengths = []

        for w in waveforms:
            wav = w  
            fb = kaldi_fbank(
                wav,
                num_mel_bins=self.num_mel_bins,
                frame_length=self.frame_length_ms,
                frame_shift=self.frame_shift_ms,
                sample_frequency=self.sample_rate,
                dither=0.0,
                energy_floor=0.0
            )
            fbanks.append(fb)
            lengths.append(fb.shape[0])

        fbanks = torch.nn.utils.rnn.pad_sequence(fbanks, batch_first=True)

        return fbanks, torch.stack(speakers), torch.stack(ages), torch.stack(genders)

# Bonus: 
def load_all_data(
    metadata_file,
    data_directory,
    batch_size=32,
    train_val_prop=0.9,   # 90/10 utterance split
    train_test_prop=0.95, # 95/05 speaker split
    speaker_subset=None, 
    min_len=2.0,
    max_len=5.0,
    sample_rate=16000,
    use_gender_sampler=False,
    seed=42
):
    df = load_voxceleb(metadata_file, data_directory)

    # 1) 95/05 split by SPEAKERS
    # -----------------------------
    speakers = df["speaker_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(speakers)

    n_total = len(speakers)
    n_trainval = max(1, int(round(train_test_prop * n_total)))  # e.g., 0.95

    trainval_speakers = set(speakers[:n_trainval])
    test_speakers = set(speakers[n_trainval:])

    trainval_df = df[df["speaker_id"].isin(trainval_speakers)].reset_index(drop=True)
    test_df = df[df["speaker_id"].isin(test_speakers)].reset_index(drop=True)
    # -----------------------------------------
    # Optional: limit number of speakers
    # speaker_subset = [N_trainval_speakers, N_test_speakers]
    # -----------------------------------------
    if speaker_subset is not None:
        n_trainval, n_test = speaker_subset

        rng2 = np.random.RandomState(seed)

        trainval_speakers_list = trainval_df["speaker_id"].unique()
        test_speakers_list = test_df["speaker_id"].unique()

        rng2.shuffle(trainval_speakers_list)
        rng2.shuffle(test_speakers_list)

        trainval_keep = set(trainval_speakers_list[:min(n_trainval, len(trainval_speakers_list))])
        test_keep = set(test_speakers_list[:min(n_test, len(test_speakers_list))])

        trainval_df = trainval_df[trainval_df["speaker_id"].isin(trainval_keep)].reset_index(drop=True)
        test_df = test_df[test_df["speaker_id"].isin(test_keep)].reset_index(drop=True)

    # -----------------------------
    # 2) 90/10 split by UTTERANCES
    # -----------------------------
    train_parts = []
    val_parts = []

    for spk, group in trainval_df.groupby("speaker_id"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        if n < 2:
            train_parts.append(group)
            continue

        n_train = max(1, int(round(train_val_prop * n)))  # e.g., 0.9
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        # ensure at least 1 in val when possible
        if len(val_idx) == 0 and n >= 2:
            val_idx = idx[-1:]
            train_idx = idx[:-1]

        train_parts.append(trainval_df.loc[train_idx])
        val_parts.append(trainval_df.loc[val_idx])

    train_df = pd.concat(train_parts, axis=0).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

    # datasets
    train_set = My_Dataset(train_df, min_len, max_len, sample_rate)
    val_set = My_Dataset(val_df, min_len, max_len, sample_rate)
    test_set = My_Dataset(test_df, min_len, max_len, sample_rate)

    # optional gender sampler (train only)
    sampler = None
    if use_gender_sampler:
        genders = train_set.df["gender"].astype(str).str.lower().str.startswith("m")
        num_male = int(genders.sum())
        num_female = int(len(genders) - num_male)
        if num_male > 0 and num_female > 0:
            weights = torch.DoubleTensor([1/num_male if g else 1/num_female for g in genders])
            sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)

    collate_fn = My_Collator(sample_rate=sample_rate,num_mel_bins=24,frame_length_ms=25.0)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,   # REQUIRED
        shuffle=False,
        collate_fn=collate_fn
    )

    # REQUIRED print
    spk_trainval = trainval_df["speaker_id"].nunique()
    spk_train = train_df["speaker_id"].nunique()
    spk_val = val_df["speaker_id"].nunique()
    spk_test = test_df["speaker_id"].nunique()
    spk_total = df["speaker_id"].nunique()

    print(
        f"Speakers | train+val: {spk_trainval} | train: {spk_train} | val: {spk_val} | test: {spk_test} | total: {spk_total}"
    )

    # IMPORTANT: return order that matches your notebook
    return test_loader, val_loader, train_loader
