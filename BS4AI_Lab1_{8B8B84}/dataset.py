import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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

# Question 7
class My_Collator:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels
        )

    def __call__(self, batch):
        waveforms, speakers, ages, genders = zip(*batch)

        mels = []
        for w in waveforms:
            mel = self.mel(w).squeeze(0)
            mels.append(mel)

        return (
            torch.stack(mels),
            torch.stack(speakers),
            torch.stack(ages),
            torch.stack(genders),
        )

# Bonus: 
def load_all_data(
    metadata_file,
    data_directory,
    batch_size=64,
    prop_train=0.8,
    prop_val=0.1,
    per_speaker=True,
    min_len=2.0,
    max_len=5.0,
    sample_rate=16000,
    use_mel=False,
):
    df = load_voxceleb(metadata_file, data_directory)

    train_df, temp_df = train_test_split(df, per_speaker, prop_train)
    val_prop = prop_val / (1.0 - prop_train)
    val_df, test_df = train_test_split(temp_df, per_speaker, val_prop)

    train_set = My_Dataset(train_df, min_len, max_len, sample_rate)
    val_set = My_Dataset(val_df, min_len, max_len, sample_rate)
    test_set = My_Dataset(test_df, min_len, max_len, sample_rate)

    genders = train_set.df["gender"].str.lower().str.startswith("m")
    num_male = genders.sum()
    num_female = len(genders) - num_male

    weights = torch.DoubleTensor(
        [1 / num_male if g else 1 / num_female for g in genders]
    )

    sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)

    collate_fn = My_Collator(sample_rate) if use_mel else None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


