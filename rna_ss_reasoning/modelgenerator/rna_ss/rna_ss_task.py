import json

from modelgenerator.tasks import *
from modelgenerator.backbones import *
from typing import Mapping, Any
from modelgenerator.rna_ss.rna_ss_utils import *
from collections import defaultdict
from modelgenerator.adapters.adapters import ResNet2DAdapter
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pickle
import torch.distributed as dist
import torch
import math

PROTENIX_EMBED_DIR = os.environ["PROTENIX_PATH"] + "/output/"
DRFOLD2_EMBED_DIR = os.environ["DRFOLD2_PATH"] + "/embeddings/"
_SHARP_LOOP_DIST_THRESHOLD = 4

def _generate_sharp_loop_mask(seq_len):
    mask = np.eye(seq_len, k=0, dtype=bool)
    for i in range(1, _SHARP_LOOP_DIST_THRESHOLD):
        mask = (
            mask + np.eye(seq_len, k=i, dtype=bool) + np.eye(seq_len, k=-i, dtype=bool)
        )

    return mask


CANONICAL_PAIRS = ["AU", "UA", "GC", "CG", "GU", "UG"]

def _generate_canonical_pairs_mask(seq: str):
    seq = seq.replace("T", "U")

    mask = np.zeros((len(seq), len(seq)), dtype=bool)
    for i, nt_i in enumerate(seq):
        for j, nt_j in enumerate(seq):
            if f"{nt_i}{nt_j}" in CANONICAL_PAIRS:
                mask[i, j] = True

    return mask

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=64, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)
    
class RNASSPairwiseTokenClassification(TaskInterface):
    """Task for fine-tuning a RNA-FM model on RNA Secondary Structure prediction.

    Args:
        backbone (BackboneCallable, optional): The callable that returns a backbone. Defaults to --model.backbone aido_dna_dummy.
        adapter (AdapterCallable, optional): The callable that returns an adapter. Defaults to LinearAdapter.
        optimizer (OptimizerCallable, optional): The optimizer to use for training. Defaults to torch.optim.AdamW.
        lr_scheduler (LRSchedulerCallable, optional): The learning rate scheduler to use for training. Defaults to None.
        batch_size (int, optional): The batch size to use for training. Defaults to None.
        use_legacy_adapter (bool, optional):
            Whether to use the adapter from the backbone. Defaults to False.
            Will be deprecated once the new adapter API is fully verified.
        strict_loading (bool, optional): Whether to strictly load the model. Defaults to True.
            Set it to False if you want to replace the adapter (e.g. for continue pretraining)
        reset_optimizer_states (bool, optional): Whether to reset the optimizer states. Defaults to False.
            Set it to True if you want to replace the adapter (e.g. for continue pretraining).
    """

    # Adapted from https://github.com/lbcb-sci/RiNALMo/blob/main/train_sec_struct_prediction.py

    def __init__(
        self,
        backbone_fn: BackboneCallable,
        adapter_fn: Optional[Callable[[int, int], SequenceAdapter]] = ResNet2DAdapter,
        tune_threshold_every_n_epoch: int = 1,
        deep_supervision: bool = True,
        unet: bool = False,
        deep_supervision_ab_trm: bool = False,
        protenix_embed: bool = True,
        drfold2_embed: bool = True,
        residual: bool = False,
        ab_triple: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.tune_threshold_every_n_epoch = tune_threshold_every_n_epoch
        self.backbone_fn = backbone_fn
        self.adapter_fn = adapter_fn
        self.backbone = None
        self.adapter = None
        self.ab_triple = ab_triple

        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.MSELoss()

        # self._eval_step_outputs = None
        self._eval_step_outputs = defaultdict(lambda: defaultdict(list))
        self._eval_step_outputs_no_relax = defaultdict(lambda: defaultdict(list))

        self._eval_step_outputs_length = None

        self.val_loss = []
        self.threshold = 0.5
        self.threshold_no_relax = 0.5
        self.THRESHOLD_TUNE_METRIC = "f1"
        self.THRESHOLD_CANDIDATES = [i / 100 for i in range(1, 30, 2)]

        self.deep_supervision = deep_supervision
        self.protenix_embed = protenix_embed
        self.drfold2_embed = drfold2_embed
        self.residual = residual

        self.pretrained_ckpt = None

        if deep_supervision:
            self.automatic_optimization = False
            self.grad_accum = 2
            self.batch_idx_count = 0

        self.deep_supervision_ab_trm = deep_supervision_ab_trm
        if deep_supervision_ab_trm:
            self.automatic_optimization = False

        self.unet = unet
        if unet:
            self.pad_emb = nn.Parameter(torch.randn(1, 1, 2048))

    # def row_col_softmax(self, y):
    #     row_softmax = torch.softmax(y, dim=-1)
    #     col_softmax = torch.softmax(y, dim=-2)
    #     return 0.5 * (row_softmax + col_softmax)

    def configure_model(self) -> None:
        if self.backbone is not None:
            return
        if self.use_legacy_adapter:
            raise NotImplementedError()
        else:
            self.backbone = self.backbone_fn(None, None)

            # YIMING
            # ckpt = torch.load("/mnt/rna01/liuzhiyuan/zyliu/yiming/rna_representation_learning-main/checkpoints/ainorna_v2_denoising/train_20250814_2246_gpu4_batchsize128_memorysize4/epoch_800.pth", map_location="cpu")
            # # ckpt = torch.load("/mnt/rna01/liuzhiyuan/zyliu/yiming/rna_representation_learning-main/checkpoints/ainorna_v2_denoising/train_20250815_1927_gpu2_batchsize128_memorysize4/epoch_1000.pth", map_location="cpu")
            
            # aino_state_dict = {k.replace("aino.", ""): v for k, v in ckpt.items() if k.startswith("aino.")}
            # self.backbone.load_state_dict(aino_state_dict, strict=False)

            if self.deep_supervision:
                self.linear_x = nn.Linear(64, 64)
                self.linear_y = nn.Linear(64, 64)
                if self.ab_triple:
                    self.linear_z = nn.Linear(64, 64)

                if self.protenix_embed:
                    if self.drfold2_embed:
                        self.drfold2_mlp = MLP(in_dim=64, hidden_dim=256)
                        self.sequence_x_linear = nn.Linear(2048, 2048)
                        self.sequence_y_linear = nn.Linear(2048, 2048)

                    self.aidorna_mlp = MLP(in_dim=4096, hidden_dim=256)
                    self.protenix_mlp = MLP(in_dim=128, hidden_dim=256)
                
                else:
                    self.in_to_conv = nn.Linear(4096, 64)

            if self.deep_supervision_ab_trm:
                self.in_to_conv = nn.Linear(4096, 64)

            self.adapter = self.adapter_fn(
                self.backbone.get_embedding_size() * 2
            )

            if self.pretrained_ckpt is not None:
                print('-------------------------------------')
                print("Loading pretrained checkpoint from", self.pretrained_ckpt)
                print('-------------------------------------')
                ckpt = torch.load(self.pretrained_ckpt, map_location="cpu")
                if self.protenix_embed:
                    self.load_state_dict(ckpt["state_dict"], strict=False)
                else:
                    self.load_state_dict(ckpt["state_dict"], strict=True)
                    
    def transform(
        self, batch: dict[str, Union[list, Tensor]], batch_idx: int
    ) -> dict[str, Union[list, Tensor]]:
        """Collates a batch of data into a format that can be passed to the forward and evaluate methods.

        Args:
            batch (dict[str, Union[list, Tensor]]): A batch of data containing sample ids, sequences, and secondary structures
            batch_idx (int): The index of the current batch in the DataLoader

        Returns:
            dict[str, Union[list, Tensor]]: The collated batch containing sample ids, sequences, secondary structures (labels), input_ids, and attention_mask
        """
        tokenized_result = self.backbone.tokenize(batch["sequences"])
        input_ids = tokenized_result.pop("input_ids", None)
        attention_mask = tokenized_result.pop("attention_mask", None)
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)
        labels = batch["sec_structures"].half().to(self.device)
        return {
            "ss_ids": batch["ss_ids"],
            "sequences": batch["sequences"],
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **tokenized_result,
        }

    def outer_concat(self, x):
        """Taken directly from FM4BioContactHead"""
        batch_size, seq_len, features = x.shape

        # Permute to [batch_size, features, seq_len]
        x = x.permute(0, 2, 1)

        # Introduce new dimensions for broadcasting
        x_1 = x[:, None, :, :, None]  # [batch_size, 1, features, seq_len, 1]
        x_2 = x[:, None, :, None, :]  # [batch_size, 1, features, 1, seq_len]

        # Repeat along new dimensions
        x_1 = x_1.repeat(
            1, 1, 1, 1, seq_len
        )  # [batch_size, 1, features, seq_len, seq_len]
        x_2 = x_2.repeat(
            1, 1, 1, seq_len, 1
        )  # [batch_size, 1, features, seq_len, seq_len]

        # Concatenate along the second dimension
        x = torch.cat((x_1, x_2), dim=1)  # [batch_size, 2, features, seq_len, seq_len]

        # Get lower triangular indices
        I, J = torch.tril_indices(seq_len, seq_len, -1)

        # Symmetrize
        x[:, :, :, I, J] = x[:, :, :, J, I]

        # Permute to desired shape and make contiguous
        x = x.permute(
            0, 3, 4, 2, 1
        ).contiguous()  # [batch_size, seq_len, seq_len, features, 2]

        # Reshape to combine the last two dimensions
        x = x.view(
            batch_size, seq_len, seq_len, features * 2
        )  # [batch_size, seq_len, seq_len, features * 2]

        return x

    # def trunc_normal_init_(self, tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    #     # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    #     # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    #     # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    #     # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    #     with torch.no_grad():
    #         if std == 0:
    #             tensor.zero_()
    #         else:
    #             sqrt2 = math.sqrt(2)
    #             a = math.erf(lower / sqrt2)
    #             b = math.erf(upper / sqrt2)
    #             z = (b - a) / 2

    #             c = (2 * math.pi) ** -0.5
    #             pdf_u = c * math.exp(-0.5 * lower ** 2)
    #             pdf_l = c * math.exp(-0.5 * upper ** 2)
    #             comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

    #             tensor.uniform_(a, b)
    #             tensor.erfinv_()
    #             tensor.mul_(sqrt2 * comp_std)
    #             tensor.clip_(lower * comp_std, upper * comp_std)

    #     return tensor

    def symmetrize_matrix(self, matrix: Tensor) -> Tensor:
        """
        Symmetrizes a square matrix by preserving the upper triangular part
        and mirroring it across the diagonal.

        Args:
            matrix (Tensor): Input tensor of shape (B, H, W), where H = W

        Returns:
            Tensor: Symmetrized tensor of shape (B, H, W).
        """
        upper_triangular = torch.triu(matrix, diagonal=1)
        return upper_triangular + upper_triangular.transpose(-1, -2)
    
    def forward(self, collated_batch: dict[str, Union[list, Tensor]]) -> Tensor:
        """Runs a forward pass of the model.

        Args:
            collated_batch (dict[str, Union[list, Tensor]]): A collated batch of data containing input_ids and attention_mask

        Returns:
            Tensor: The classifier logits
        """

        no_cycling = getattr(self, "no_cycling", 5)
        k_no_cycling = getattr(self, "k_no_cycling", 2)

        if self.use_legacy_adapter:
            raise NotImplementedError()
        else:
            if self.deep_supervision:
                logits = []   
                for step in range(4):
                    encoder_hidden = self.backbone(**collated_batch) # seq_len * 2048
                    x = self.outer_concat(encoder_hidden) # [batch_size, seq_len, seq_len, 2048 * 2]
                    x = x[:, 1:-1, 1:-1]
                    cls_token = encoder_hidden[:, 0:1, :] # [batch_size, 1, 2048]
                    encoder_hidden = encoder_hidden[:, 1:-1, :]
                    if self.protenix_embed:
                        x = self.aidorna_mlp(x) # [batch_size, seq_len, seq_len, 64]
                        protenix_embedding = torch.load(f"{PROTENIX_EMBED_DIR}/{collated_batch['ss_ids'][0]}.pt", map_location="cpu").to(x.device, dtype=x.dtype).unsqueeze(0)
                        if self.drfold2_embed:
                            drfold2_embedding = torch.load(f"{DRFOLD2_EMBED_DIR}/{collated_batch['ss_ids'][0]}.pt", map_location="cpu").to(x.device, dtype=x.dtype).unsqueeze(0)
                            # print({collated_batch['ss_ids'][0]})
                            # print(f"drfold2_embedding shape: {drfold2_embedding.shape}")
                            # print(f"protenix_embedding shape: {protenix_embedding.shape}")
                            # print(f"x shape: {x.shape}")
                            x = x + self.protenix_mlp(protenix_embedding) + self.drfold2_mlp(drfold2_embedding)
                        else:
                            x = x + self.protenix_mlp(protenix_embedding)
                    else:
                        x = self.in_to_conv(x)

                    if step == 0:
                        # x_recycle = torch.zeros_like(x, device=x.device)
                        # y_recycle = torch.zeros_like(x, device=x.device)
                        x_recycle = self.linear_x(x)
                        y_recycle = self.linear_y(x)
                        if self.ab_triple:
                            z_recycle = self.linear_z(x)
                    if self.drfold2_embed:
                        sequence_x = self.sequence_x_linear(encoder_hidden) # linear 2048 -> 2048
                        sequence_y = self.sequence_y_linear(encoder_hidden) # linear 2048 -> 2048
                        if self.ab_triple:
                            cls_logits, x_recycle, y_recycle, z_recycle = self.adapter(x, no_cycling=no_cycling, k_no_cycling=k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle, z_recycle=z_recycle, sequence_x=sequence_x, sequence_y=sequence_y)
                        else:
                            cls_logits, x_recycle, y_recycle = self.adapter(x, no_cycling=no_cycling, k_no_cycling=k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle, sequence_x=sequence_x, sequence_y=sequence_y)
                    else:
                        # cls_logits, x_recycle = self.adapter(x, no_cycling=no_cycling, k_no_cycling=k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle)
                        cls_logits, x_recycle, y_recycle = self.adapter(x, no_cycling=no_cycling, k_no_cycling=k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle)
                    
                    logits.append(cls_logits)
                    
                return logits

            elif self.deep_supervision_ab_trm:
                for step in range(4):
                    encoder_hidden = self.backbone(**collated_batch)
                    x = self.outer_concat(encoder_hidden) # [batch_size, seq_len, seq_len, features * 2]
                    x = x[:, 1:-1, 1:-1]
                    x = self.in_to_conv(x)
                    if step == 0:
                        # x_recycle = self.trunc_normal_init_(
                        #     torch.empty_like(x, device=x.device),
                        #     std=1.0
                        # )
                        # y_recycle = self.trunc_normal_init_(
                        #     torch.empty_like(x, device=x.device),
                        #     std=1.0
                        # )
                        # print(x.dtype) # torch.float16
                        x_recycle = torch.empty_like(x, device=x.device, dtype=torch.float32)
                        x_recycle.normal_(mean=0.0, std=1.0)
                        x_recycle = x_recycle.clamp(-2.0, 2.0)
                        x_recycle = x_recycle.to(x.dtype)

                        y_recycle = torch.empty_like(x, device=x.device, dtype=torch.float32)
                        y_recycle.normal_(mean=0.0, std=1.0)
                        y_recycle = y_recycle.clamp(-2.0, 2.0)
                        y_recycle = y_recycle.to(x.dtype)

                        # print("x_recycle:", x_recycle)
                        # print("y_recycle:", y_recycle)

                    cls_logits, x_recycle, y_recycle = self.adapter(x, no_cycling=no_cycling, k_no_cycling=k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle)
                    # print(cls_logits.shape)

            else:    
                encoder_hidden = self.backbone(**collated_batch)
                if self.unet:
                    B, L, D = encoder_hidden.shape
                    l = L - 2
                    target_len = ((l - 1) // 16 + 1) * 16
                    pad_len = target_len - l
                    if pad_len > 0:
                        pad_emb = self.pad_emb.repeat(B, pad_len, 1)
                        encoder_hidden = torch.cat([encoder_hidden[:, 0:1, :], encoder_hidden[:, 1:-1, :], pad_emb, encoder_hidden[:, -1:, :]], dim=1)
                    x = self.outer_concat(encoder_hidden)
                    cls_logits = self.adapter(x[:, 1:-1, 1:-1], no_cycling=no_cycling, k_no_cycling=k_no_cycling, seq=collated_batch["sequences"][0])
                    if pad_len > 0:
                        cls_logits = cls_logits[:, :l, :l]

                else:
                    x = self.outer_concat(encoder_hidden)
                    cls_logits = self.adapter(x[:, 1:-1, 1:-1], no_cycling=no_cycling, k_no_cycling=k_no_cycling, seq=collated_batch["sequences"][0])
        
        return cls_logits

    def evaluate(
        self,
        logits,
        collated_batch: dict[str, Union[list, Tensor]],
        stage: Optional[Literal["train", "val", "test"]] = None,
        loss_only: bool = False,
        train_rule: bool = False
    ) -> dict[str, Union[Tensor, float]]:
        """Evaluates the model predictions against the ground truth labels.

        Args:
            logits (Tensor): The sequence-level model predictions
            collated_batch (dict[str, Union[list, Tensor]]): The collated batch of data containing labels
            stage (Optional[str], optional): The stage of training (train, val, test). Required if loss_only is False.
            loss_only (bool, optional): Whether to only return the loss. Defaults to False.

        Returns:
            Tensor: The loss.
        """
        labels = collated_batch["labels"]
        # print(labels.shape)

        seq = collated_batch["sequences"][0]
        # print(len(seq)) # 1518
        # print('-------------------')
        # print(collated_batch["ss_ids"])
        # print('-------------------')
        # print(logits.shape) # torch.Size([1, 1022, 1022])

        seq_len = labels.shape[1]

        if train_rule:
            allowed_pairs_mask = np.logical_not(np.eye(seq_len, dtype=bool))

            # Prevent pairings that would cause sharp loops
            allowed_pairs_mask = np.logical_and(
                allowed_pairs_mask, ~_generate_sharp_loop_mask(seq_len)
            )

            # Prevent non-canonical pairings
            allowed_pairs_mask = np.logical_and(
                allowed_pairs_mask, _generate_canonical_pairs_mask(seq)
            )

            allowed_pairs_mask = torch.tensor(allowed_pairs_mask, dtype=torch.bool, device=logits.device)
            allowed_pairs_mask = allowed_pairs_mask.unsqueeze(0)  # shape: (1, seq_len, seq_len)
            
            logits = logits.masked_fill(~allowed_pairs_mask, -1000.0)

        if isinstance(logits, list):
            upper_tri_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=logits[-1].device),
                diagonal=1,
            )
            for index, logit in enumerate(logits):
                if index == 0:
                    loss = self.loss(
                        logit[..., upper_tri_mask], labels[..., upper_tri_mask].to(logit)
                    )
                else:
                    loss += self.loss(
                        logit[..., upper_tri_mask], labels[..., upper_tri_mask].to(logit)
                    )
            loss = loss / len(logits)
        else:
            upper_tri_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=logits.device),
                diagonal=1,
            )
            loss = self.loss(
                logits[..., upper_tri_mask], labels[..., upper_tri_mask].to(logits)
            )
        if loss_only:
            return {"loss": loss}
        
        metrics = self.get_metrics_by_stage(stage)

        if isinstance(logits, list):
            preds = logits[-1].argmax(-1)
        else:
            preds = logits.argmax(-1)
        
        for metric in metrics.values():
            metric(preds, labels)
        return {"loss": loss}

    # def on_train_start(self):
    #     for i, lora in enumerate(self.adapter.loras):
    #         total_params = sum(p.numel() for p in lora.parameters())
    #         trainable_params = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    #         print(f"[on_train_start] LoRA {i}: total params = {total_params}, trainable params = {trainable_params}")

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            trainable_param = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            self.log("TrPara", trainable_param, prog_bar=True)

        # self.no_cycling = random.randint(1, 5)
        # self.k_no_cycling = random.randint(0, 4)

        self.no_cycling = 5
        self.k_no_cycling = 2

        if self.deep_supervision:
            opt = self.optimizers()

            collated_batch = self.transform(batch, batch_idx)

            for step in range(4):
                encoder_hidden = self.backbone(**collated_batch)
                cls_token = encoder_hidden[:, 0:1, :] # [batch_size, 1, 2048]
                x = self.outer_concat(encoder_hidden) # [batch_size, seq_len, seq_len, features * 2]

                x = x[:, 1:-1, 1:-1]
                encoder_hidden = encoder_hidden[:, 1:-1, :]
                if self.protenix_embed:
                    x = self.aidorna_mlp(x)
                    protenix_embedding = torch.load(f"{PROTENIX_EMBED_DIR}/{collated_batch['ss_ids'][0]}.pt", map_location="cpu").to(x.device).unsqueeze(0)
                    if self.drfold2_embed:
                        drfold2_embedding = torch.load(f"{DRFOLD2_EMBED_DIR}/{collated_batch['ss_ids'][0]}.pt", map_location="cpu").to(x.device, dtype=x.dtype).unsqueeze(0)
                        x = x + self.protenix_mlp(protenix_embedding) + self.drfold2_mlp(drfold2_embedding)
                    else:
                        x = x + self.protenix_mlp(protenix_embedding)
                else:
                    x = self.in_to_conv(x) # 4096 -> 64

                if step == 0:
                    # x_recycle = torch.zeros_like(x, device=x.device)
                    # y_recycle = torch.zeros_like(x, device=x.device)
                    x_recycle = self.linear_x(x)
                    y_recycle = self.linear_y(x)
                    if self.ab_triple:
                        z_recycle = self.linear_z(x)

                if self.drfold2_embed:
                    sequence_x = self.sequence_x_linear(encoder_hidden) # linear 2048 -> 2048
                    sequence_y = self.sequence_y_linear(encoder_hidden) # linear 2048 -> 2048
                    if self.ab_triple:
                        cls_logits, x_recycle, y_recycle, z_recycle = self.adapter(x, no_cycling=self.no_cycling, k_no_cycling=self.k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle, z_recycle=z_recycle, sequence_x=sequence_x, sequence_y=sequence_y)
                    else:
                        cls_logits, x_recycle, y_recycle = self.adapter(x, no_cycling=self.no_cycling, k_no_cycling=self.k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle, sequence_x=sequence_x, sequence_y=sequence_y)
                else:
                    # cls_logits, x_recycle = self.adapter(x, no_cycling=self.no_cycling, k_no_cycling=self.k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle)
                    cls_logits, x_recycle, y_recycle = self.adapter(x, no_cycling=self.no_cycling, k_no_cycling=self.k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle)
                
                outputs = self.evaluate(cls_logits, collated_batch, "train", loss_only=False)
                loss = outputs["loss"]
                self.log_loss_and_metrics(loss, "train")
                self.manual_backward(loss / 4)
            
            self.batch_idx_count += 1

            if self.batch_idx_count % self.grad_accum == 0:
                scheduler = self.lr_schedulers()
                opt.step()
                opt.zero_grad()
                scheduler.step()

            return outputs
        
        if self.deep_supervision_ab_trm:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            collated_batch = self.transform(batch, batch_idx)

            for step in range(4):
                encoder_hidden = self.backbone(**collated_batch)
                x = self.outer_concat(encoder_hidden) # [batch_size, seq_len, seq_len, features * 2]

                x = x[:, 1:-1, 1:-1]
                x = self.in_to_conv(x)

                if step == 0:
                    # x_recycle = self.trunc_normal_init_(
                    #     torch.empty_like(x, device=x.device),
                    #     std=1.0
                    # )
                    # y_recycle = self.trunc_normal_init_(
                    #     torch.empty_like(x, device=x.device),
                    #     std=1.0
                    # )
                    x_recycle = torch.empty_like(x, device=x.device, dtype=torch.float32)
                    x_recycle.normal_(mean=0.0, std=1.0)
                    x_recycle = x_recycle.clamp(-2.0, 2.0)
                    x_recycle = x_recycle.to(x.dtype)

                    y_recycle = torch.empty_like(x, device=x.device, dtype=torch.float32)
                    y_recycle.normal_(mean=0.0, std=1.0)
                    y_recycle = y_recycle.clamp(-2.0, 2.0)
                    y_recycle = y_recycle.to(x.dtype)

                cls_logits, x_recycle, y_recycle = self.adapter(x, no_cycling=self.no_cycling, k_no_cycling=self.k_no_cycling, seq=collated_batch["sequences"][0], x_recycle=x_recycle, y_recycle=y_recycle)
                outputs = self.evaluate(cls_logits, collated_batch, "train", loss_only=False)
                loss = outputs["loss"]
                self.log_loss_and_metrics(loss, "train")
                self.manual_backward(loss)
                opt.step()
                opt.zero_grad()
                scheduler.step()

            return outputs
        
        return super().training_step(batch, batch_idx)

    def on_validation_start(self):
        self._reset_eval_step_outputs()
        self.save_dict = {
            "val": {},
            "test": {},
            "threshold": 0.0
        }
        self.save_ds = []

    def validation_step(self, batch, batch_idx):
        self.no_cycling = 5
        self.k_no_cycling = 2
        return self._eval_step(
            batch, batch_idx, thresholds=self.THRESHOLD_CANDIDATES, log_prefix="val"
        )

    def on_validation_epoch_end(self):
        if (
            self.trainer.sanity_checking
            or (self.trainer.current_epoch + 1) % self.tune_threshold_every_n_epoch != 0
        ):
            # If sanity checking, don't update the threshold
            return

        # Find threshold with highest validation score
        best_metric_val = -1.0
        best_threshold = 0.0

        # need to comment if no need to relax
        # for threshold in self.THRESHOLD_CANDIDATES:
        #     curr_metric_val = sum(
        #         self._eval_step_outputs[self.THRESHOLD_TUNE_METRIC][threshold]
        #     ) / len(self._eval_step_outputs[self.THRESHOLD_TUNE_METRIC][threshold])

        #     if curr_metric_val > best_metric_val:
        #         best_metric_val = curr_metric_val
        #         best_threshold = threshold

        self.threshold = best_threshold

        best_metric_val_no_relax = -1.0
        best_threshold_no_relax = 0.0
        for threshold in self.THRESHOLD_CANDIDATES:
            curr_metric_val_no_relax = sum(
                self._eval_step_outputs_no_relax[self.THRESHOLD_TUNE_METRIC][threshold]
            ) / len(self._eval_step_outputs_no_relax[self.THRESHOLD_TUNE_METRIC][threshold])

            if curr_metric_val_no_relax > best_metric_val_no_relax:
                best_metric_val_no_relax = curr_metric_val_no_relax
                best_threshold_no_relax = threshold

        self.threshold_no_relax = best_threshold_no_relax

        self.log(f"val_f1", best_metric_val, prog_bar=True, sync_dist=True)
        self.log(f"thresh", self.threshold, prog_bar=True, sync_dist=True)
        self.log("val_f1_no_relax", best_metric_val_no_relax, prog_bar=True, sync_dist=True)
        self.log("thresh_no_relax", self.threshold_no_relax, prog_bar=True, sync_dist=True)

        self.log_dict(
            {
                f"val/{self.THRESHOLD_TUNE_METRIC}": best_metric_val,
                f"val/{self.THRESHOLD_TUNE_METRIC}_no_relax": best_metric_val_no_relax,
                f"val/threshold": self.threshold,
                f"val/threshold_no_relax": self.threshold_no_relax,
                "val/loss": sum(self.val_loss) / len(self.val_loss),
            },
            sync_dist=True,
        )

        # if self.trainer.current_epoch >= 0:
        #     self.trainer.test(model=self, datamodule=self)

        # yiming
        self._reset_eval_step_outputs()
        test_loader = self.trainer.datamodule.test_dataloader()

        for batch in tqdm(test_loader, desc="Testing: ", total=len(test_loader)):
            self.test_step(batch, 0)

        # for key in self._eval_step_outputs:
        #     metric_avg_val = sum(self._eval_step_outputs[key][self.threshold]) / len(
        #         self._eval_step_outputs[key][self.threshold]
        #     )
        #     self.log(f"test/{key.lower()}", metric_avg_val, sync_dist=True)
        
        for key in self._eval_step_outputs_no_relax:
            metric_avg_val = sum(self._eval_step_outputs_no_relax[key][self.threshold_no_relax]) / len(
                self._eval_step_outputs_no_relax[key][self.threshold_no_relax]
            )
            self.log(f"test/{key.lower()}_no_relax", metric_avg_val, sync_dist=True)
        ###

    def on_test_start(self):
        self._reset_eval_step_outputs()
        self.save_dict = {
            "val": {},
            "test": {},
            "threshold": 0.0
        }
        self.save_ds = []

    def test_step(self, batch, batch_idx):
        self.no_cycling = 5
        self.k_no_cycling = 2
        return self._eval_step(
            batch, batch_idx, thresholds=[self.threshold, self.threshold_no_relax], log_prefix="test"
        )

    def on_test_epoch_end(self):
        # Get macro average of each metric
        # for key in self._eval_step_outputs:
        #     metric_avg_val = sum(self._eval_step_outputs[key][self.threshold]) / len(
        #         self._eval_step_outputs[key][self.threshold]
        #     )
        #     self.log(f"test/{key.lower()}", metric_avg_val)
        
        for key in self._eval_step_outputs_no_relax:
            metric_avg_val = sum(self._eval_step_outputs_no_relax[key][self.threshold_no_relax]) / len(
                self._eval_step_outputs_no_relax[key][self.threshold_no_relax]
            )
            self.log(f"test/{key.lower()}", metric_avg_val)

    def _update_eval_step_outputs(
        self, logits, sec_struct_true, ss_ids, seqs, thresholds, log_prefix
    ):

        if isinstance(logits, list):
            batch_size, *_ = logits[-1].shape
            probs = torch.mean(
                torch.stack([torch.sigmoid(x) for x in logits], dim=0),
                dim=0
            )
        else:
            batch_size, *_ = logits.shape
            probs = torch.sigmoid(logits)
            # probs = logits.clone()

        if probs.dtype == torch.bfloat16:
            # Cast brain floating point into floating point
            probs = probs.type(torch.float16)

        probs = probs.cpu().numpy()
        sec_struct_true = sec_struct_true.cpu().numpy()

        if log_prefix == "test":
            self.save_dict["threshold"] = thresholds[0]

            for i in range(batch_size):
                y_true = sec_struct_true[i]

                self.save_ds.append({
                    "ss_id": ss_ids[i],
                    "seq": seqs[i],
                    "probs": probs[i],
                    "gt": y_true
                })

                # need to comment if no need to relax
                # with relax
                # sec_struct_pred = prob_mat_to_sec_struct(
                #     probs=probs[i], seq=seqs[i], threshold=thresholds[0]
                # )

                # y_pred = sec_struct_pred

                # self._eval_step_outputs["precision"][thresholds[0]].append(
                #     ss_precision(y_true, y_pred)
                # )
                # self._eval_step_outputs["recall"][thresholds[0]].append(
                #     ss_recall(y_true, y_pred)
                # )
                # self._eval_step_outputs["f1"][thresholds[0]].append(ss_f1(y_true, y_pred))

                # without relax
                sec_struct_pred = prob_mat_to_sec_struct(
                    probs=probs[i], seq=seqs[i], threshold=thresholds[1]
                )

                y_pred = sec_struct_pred

                self._eval_step_outputs_no_relax["precision"][thresholds[1]].append(
                    ss_precision(y_true, y_pred, allow_flexible_pairings=False)
                )
                self._eval_step_outputs_no_relax["recall"][thresholds[1]].append(
                    ss_recall(y_true, y_pred, allow_flexible_pairings=False)
                )
                self._eval_step_outputs_no_relax["f1"][thresholds[1]].append(ss_f1(y_true, y_pred, allow_flexible_pairings=False))

                if self.trainer.testing:
                    save_path = os.environ["SAVE_PATH"]
                    os.makedirs(save_path, exist_ok=True)

                    save_to_ct(
                        # Path(save_path) / f"{ss_ids[i]}_pred_f1={f1_score}.ct",
                        Path(save_path) / f"{ss_ids[i]}.ct",
                        sec_struct=y_pred,
                        seq=seqs[i],
                    )
                
        else:
            for i in range(batch_size):            
                for threshold in thresholds:
                    sec_struct_pred = prob_mat_to_sec_struct(
                        probs=probs[i], seq=seqs[i], threshold=threshold
                    )

                    y_true = sec_struct_true[i]
                    y_pred = sec_struct_pred

                    self._eval_step_outputs_no_relax["precision"][threshold].append(
                        ss_precision(y_true, y_pred, allow_flexible_pairings=False)
                    )
                    self._eval_step_outputs_no_relax["recall"][threshold].append(
                        ss_recall(y_true, y_pred, allow_flexible_pairings=False)
                    )
                    self._eval_step_outputs_no_relax["f1"][threshold].append(
                        ss_f1(y_true, y_pred, allow_flexible_pairings=False)
                    )

    def _reset_eval_step_outputs(self):
        torch.cuda.empty_cache()
        self._eval_step_outputs.clear()
        self._eval_step_outputs_no_relax.clear()
        self.val_loss.clear()

    def _eval_step(self, batch, batch_idx, thresholds, log_prefix):
        collated_batch = self.transform(batch, batch_idx)
        ss_ids = collated_batch["ss_ids"]
        seqs = collated_batch["sequences"]
        sec_struct_true = collated_batch["labels"]
        logits = self(collated_batch)

        # if isinstance(logits, list):
        #     logits = logits[-1]

        if log_prefix == 'val':
            val_loss = self.evaluate(logits, collated_batch, log_prefix, loss_only=True)["loss"]
            self.val_loss.append(val_loss.cpu().item())

        if (
            self.trainer.testing
            or (self.trainer.current_epoch + 1) % self.tune_threshold_every_n_epoch == 0
        ):
            self._update_eval_step_outputs(
                logits, sec_struct_true, ss_ids, seqs, thresholds, log_prefix
            )

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = dict(state_dict)
        # print("-------------------------------------")
        # print(state_dict.keys())
        # print("-------------------------------------")
        
        # self.threshold = state_dict["threshold"]
        self.threshold_no_relax = state_dict["threshold"]

        state_dict.pop(
            "threshold"
        )  # Remove 'threshold' key for possible "strict" clashes

        # opt = self.optimizers()
        # if "optimizer_states" in state_dict:
        #     opt.load_state_dict(state_dict["optimizer_states"][0])

        return super().load_state_dict(state_dict, strict, assign)

    def _sync_mean(self, value):
        """Sync a scalar (float or 0-d tensor) across DDP ranks by mean."""
        if not dist.is_available() or not dist.is_initialized():
            return value

        t = torch.tensor(value, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
        return t.item()
    
    def on_save_checkpoint(self, checkpoint):
        # default relax threshold
        # save_threshold = self.threshold
        save_threshold = self._sync_mean(self.threshold)

        for cb in self.trainer.callbacks:
            if hasattr(cb, "monitor") and cb.monitor == "val/f1_no_relax":
                save_threshold = self._sync_mean(self.threshold_no_relax)
                # save_threshold = self.threshold_no_relax
                break

        checkpoint["state_dict"]["threshold"] = save_threshold

        return super().on_save_checkpoint(checkpoint)