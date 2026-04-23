import torch
import numpy as np
from torch.utils.data import Dataset
import convminds as cm
from convminds.benchmarks import HuthBenchmark
from convminds.transforms.pca import VoxelPCA
from convminds.data.primitives import BrainTensor

class HuthAlignmentDataset(Dataset):
    """
    Standardized dataset for Brain-to-Language alignment using the Huth protocol.
    
    Windowing logic:
    - Target Words: TR X
    - Story Context: TR X-3 to X-1
    - Brain Input: BOLD TR X+1 to X+4 (capturing HRF lag)
    """
    def __init__(self, subject_ids=["S1"], split="train", pca_dim=1000, tr_window=(1, 5)):
        self.bold_data = {}
        self.story_metadata = {}
        self.all_samples = []
        self.tr_window = tr_window
        
        # Load and preprocess
        for subj in subject_ids:
            # 1. Initialize benchmark for this subject (ensures data materialization)
            benchmark = HuthBenchmark(subject=subj)
            
            # 2. Load Raw Recordings
            recording_data = benchmark.human_recording_source.load_recordings(
                benchmark, selector={"subject": subj}
            )
            
            # 2. Setup PCA
            all_bold = np.vstack(recording_data.values)
            # Use a subject-specific cache path
            cache_path = cm.cache.convminds_home() / "cache" / "pca" / f"huth_{subj}_pca_{pca_dim}.joblib"
            pca = VoxelPCA(n_components=pca_dim, cache_path=cache_path)
            
            # Wrap for PCA fitting
            brain_for_pca = BrainTensor(
                torch.from_numpy(all_bold).unsqueeze(0).float(),
                torch.zeros((all_bold.shape[1], 3)),
                recording_data.metadata.get("rois", {})
            )
            pca.fit(brain_for_pca)
            
            # 3. Process each story
            self.bold_data[subj] = {}
            for story_idx, story_name in enumerate(recording_data.stimulus_ids):
                # Project
                story_bold = recording_data.values[story_idx]
                bt = BrainTensor(torch.from_numpy(story_bold).unsqueeze(0).float(), torch.zeros(story_bold.shape[1], 3), {})
                projected = pca(bt).signal.squeeze(0).numpy()
                
                # Z-score normalization per run
                mean = projected.mean(axis=0, keepdims=True)
                std = projected.std(axis=0, keepdims=True) + 1e-8
                projected = (projected - mean) / std
                
                self.bold_data[subj][story_name] = projected
                
                # Get story metadata for alignment
                # benchmark.stimuli contains records with metadata
                record = None
                for r in benchmark.stimuli:
                    if r.stimulus_id == story_name:
                        record = r
                        break
                
                if record is None: continue
                self.story_metadata[story_name] = record.metadata
                
                actual_trs = projected.shape[0]
                test_start = int(actual_trs * 0.9)
                tr_range = range(3, actual_trs - 4)
                if split == "train":
                    tr_range = range(3, test_start - 4)
                elif split == "test":
                    tr_range = range(test_start, actual_trs - 4)
                
                for x in tr_range:
                    self.all_samples.append((subj, story_name, x))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, index):
        subj, story, x = self.all_samples[index]
        w_start, w_end = self.tr_window
        bold_window = self.bold_data[subj][story][x+w_start:x+w_end]
        
        metadata = self.story_metadata[story]
        tr_times = metadata["tr_times"]
        word_intervals = metadata["word_intervals"]
        
        def get_words(tr_start, tr_end):
            t_start = tr_times[tr_start]
            t_end = tr_times[tr_end+1] if tr_end+1 < len(tr_times) else tr_times[-1] + 2.0
            noise_tokens = {"sp", "uh", "um"}
            words = [i["text"].lower().strip() for i in word_intervals 
                     if i["xmin"] >= t_start and i["xmin"] < t_end]
            words = [w for w in words if w not in noise_tokens and len(w) > 0]
            return " ".join(words) if words else ""

        return {
            "bold": torch.from_numpy(bold_window).float(),
            "context": get_words(x-3, x-1),
            "target": get_words(x, x+1),
            "subject": subj,
            "story": story,
            "tr": x,
            "time_window": (tr_times[x], tr_times[x+2] if x+2 < len(tr_times) else tr_times[x]+4.0)
        }
