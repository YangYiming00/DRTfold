from modelgenerator.data import *
from typing import Union, Optional
from pathlib import Path
import _pickle as cPickle
import collections
from Bio import SeqIO
from modelgenerator.rna_ss.rna_ss_utils import parse_sec_struct_file


class RNASSPairwiseTokenClassification(DataInterface):

    SUPPORTED_DATASETS = [
        "bpRNA",
        "archiveII",
        "PDB-TS1",
        "PDB-TS2",
        "PDB-TS3",
        "Rfam15.0-15.1",
        "bpRNA-new",
        "PDB2607"
    ]

    def __init__(
        self,
        *args,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        min_seq_len: int = 0,
        max_seq_len: int = 700,
        dataset: str = "bpRNA",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dataset = dataset
        self.data_root = Path(self.path)

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

        self.batch_size = 1

        if dataset == "bpRNA":
            self.train_dir = "bpRNA"
            self.val_dir = "bpRNA"
            self.test_dir = "bpRNA"

        elif dataset == "archiveII":
            self.train_dir = "archiveII"
            self.val_dir = "archiveII"
            self.test_dir = "archiveII"

        elif dataset == "PDB-TS1":
            self.train_dir = "PDB-TS1"
            self.val_dir = "PDB-TS1"
            self.test_dir = "PDB-TS1"

        elif dataset == "PDB-TS2":
            self.train_dir = "PDB-TS2"
            self.val_dir = "PDB-TS2"
            self.test_dir = "PDB-TS2"

        elif dataset == "PDB-TS3":
            self.train_dir = "PDB-TS3"
            self.val_dir = "PDB-TS3"
            self.test_dir = "PDB-TS3"

        elif dataset == "Rfam15.0-15.1":
            self.train_dir = "Rfam15.0-15.1"
            self.val_dir = "Rfam15.0-15.1"
            self.test_dir = "Rfam15.0-15.1"

        elif dataset == "bpRNA-new":
            self.train_dir = "bpRNA-new"
            self.val_dir = "bpRNA-new"
            self.test_dir = "bpRNA-new"

        elif dataset == "PDB2607":
            self.train_dir = "PDB2607"
            self.val_dir = "PDB2607"
            self.test_dir = "PDB2607"

        elif dataset.endswith("fasta"):
            self.train_dir = dataset
            self.val_dir = dataset
            self.test_dir = dataset

        else:
            raise NotImplementedError(
                f"Dataset '{dataset}' is currently not supported! Please use one of the following: {self.SUPPORTED_DATASETS}"
            )

    def prepare_dataset(self, data_dir):
        ss_ids = []
        sequences = []
        sec_structures = []

        for record in SeqIO.parse(data_dir, "fasta"):
            seq_id = record.id
            seq = str(record.seq)
            seq = seq.upper().replace("T", "U")
            seq = re.sub(r"[^AUCG]", "N", seq)
            L = len(seq)

            assert (
                L <= 1022
            ), "dataset should not have sequences longer than 1022 tokens."

            ss = np.zeros((L, L), dtype=np.float32)

            ss_ids.append(seq_id)
            sequences.append(seq)
            sec_structures.append(ss)
        
        return ss_ids, sequences, sec_structures
    
    def setup(self, stage: Optional[str] = None):
        tmp = self.prepare_dataset(
            # data_dir=self.data_root / self.train_dir,
            data_dir=self.train_dir,
            # min_seq_len=self.min_seq_len,
            # max_seq_len=500,
        )
        # self.train_dataset = AnyDataset(
        #     ss_ids=tmp[0], sequences=tmp[1], sec_structures=tmp[2], train_mode=True
        # )
        self.train_dataset = AnyDataset(
            ss_ids=tmp[0], sequences=tmp[1], sec_structures=tmp[2]
        )

        tmp = self.prepare_dataset(
            # data_dir=self.data_root / self.val_dir,
            data_dir=self.val_dir,
            # min_seq_len=self.min_seq_len,
            # max_seq_len=700
        )
        # self.val_dataset = AnyDataset(
        #     ss_ids=tmp[0], sequences=tmp[1], sec_structures=tmp[2], train_mode=False
        # )
        self.val_dataset = AnyDataset(
            ss_ids=tmp[0], sequences=tmp[1], sec_structures=tmp[2]
        )
        tmp = self.prepare_dataset(
            # data_dir=self.data_root / self.test_dir,
            data_dir=self.test_dir,
            # min_seq_len=self.min_seq_len,
            # max_seq_len=1022
        )
        # self.test_dataset = AnyDataset(
        #     ss_ids=tmp[0], sequences=tmp[1], sec_structures=tmp[2], train_mode=False
        # )
        self.test_dataset = AnyDataset(
            ss_ids=tmp[0], sequences=tmp[1], sec_structures=tmp[2]
        )
