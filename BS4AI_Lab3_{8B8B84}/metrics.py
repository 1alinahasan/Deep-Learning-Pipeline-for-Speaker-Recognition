# metrics.py
import numpy as np

def cosine_scoring_all(embeddings, labels):
    
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    E = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels)

    N = E.shape[0]

    if y.shape[0] != N:
        raise ValueError("Labels length does not match embeddings rows")

    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    E = E / norms

    S = E @ E.T
    np.fill_diagonal(S, 1.0)

    same = (y[:, None] == y[None, :]).astype(np.int32)

    mask = ~np.eye(N, dtype=bool)

    scores_list = S[mask].astype(np.float32).tolist()
    pair_labels = same[mask].astype(np.int32).tolist()

    expected = N * N - N
    if len(scores_list) != expected:
        raise RuntimeError("Output size mismatch")

    return scores_list, pair_labels


def FAR_FRR(scores, targets, threshold=0.5):

    s = np.asarray(scores, dtype=np.float32)
    t = np.asarray(targets, dtype=np.int32)

    accept = s >= threshold

    non_mask = (t == 0)
    n_non = int(non_mask.sum())
    FAR = (accept[non_mask].sum() / n_non) if n_non > 0 else 0.0

    tar_mask = (t == 1)
    n_tar = int(tar_mask.sum())
    FRR = ((~accept)[tar_mask].sum() / n_tar) if n_tar > 0 else 0.0

    return float(FAR), float(FRR)


def FAR_FRR_all(scores, targets, N=1000):

    thresholds = np.linspace(0.0, 1.0, N, dtype=np.float32)
    FARs = []
    FRRs = []
    for th in thresholds:
        far, frr = FAR_FRR(scores, targets, threshold=float(th))
        FARs.append(far)
        FRRs.append(frr)
    return thresholds, np.array(FARs, dtype=np.float32), np.array(FRRs, dtype=np.float32)


def EER(scores, targets):
   
    s = np.asarray(scores)
    t = np.asarray(targets)

    idx = np.argsort(-s)
    s = s[idx]
    t = t[idx]

    P = np.sum(t == 1)
    N = np.sum(t == 0)

    tar_cum = np.cumsum(t == 1)
    non_cum = np.cumsum(t == 0)

    FAR = non_cum / N

    FRR = (P - tar_cum) / P

    diff = np.abs(FAR - FRR)
    idx_eer = np.argmin(diff)

    eer = (FAR[idx_eer] + FRR[idx_eer]) / 2
    threshold = s[idx_eer]

    return float(eer), float(threshold)



def cosine_scoring_from_trials(test_embeddings, test_utt, trials_df):
    
    if "torch" in str(type(test_embeddings)):
        test_embeddings = test_embeddings.detach().cpu().numpy()
    if "torch" in str(type(test_utt)):
        test_utt = test_utt.detach().cpu().numpy()

    E = np.asarray(test_embeddings, dtype=np.float32)
    utt = np.asarray(test_utt).astype(str)

    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    E = E / norms

    utt_to_idx = {u: i for i, u in enumerate(utt)}

    cols = list(trials_df.columns)

    if "label" in cols:
        targets = trials_df["label"].astype(int).values
    elif "targettype" in cols:
        targets = (trials_df["targettype"].astype(str).str.lower() == "target").astype(int).values
    else:
        targets = trials_df.iloc[:, 0].astype(int).values

    if "utt1" in cols and "utt2" in cols:
        u1 = trials_df["utt1"].astype(str).values
        u2 = trials_df["utt2"].astype(str).values

        if "/" in u1[0] or "\\" in u1[0]:
            def path_to_id(p):
                p = p.replace("\\", "/")
                base = p.split("/")[-1]              # 00003.wav
                base = os.path.splitext(base)[0]     # 00003
                return p  
            pass

    elif "modelid" in cols and "segmentid" in cols:
        u1 = trials_df["modelid"].astype(str).values
        u2 = trials_df["segmentid"].astype(str).values
    else:
        u1 = trials_df.iloc[:, 1].astype(str).values
        u2 = trials_df.iloc[:, 2].astype(str).values

    scores = np.empty(len(u1), dtype=np.float32)

    for i in range(len(u1)):
        a = u1[i]
        b = u2[i]
        ia = utt_to_idx.get(a, None)
        ib = utt_to_idx.get(b, None)
        if ia is None or ib is None:
            scores[i] = 0.0
        else:
            scores[i] = float(np.dot(E[ia], E[ib]))

    return scores, targets

