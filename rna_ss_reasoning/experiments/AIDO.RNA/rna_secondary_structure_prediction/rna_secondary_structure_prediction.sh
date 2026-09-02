# Sazan Mahbub
MODE=$1 ## set it to "train" for finetuning the RNA-FM for RNA secondary structure prediction

RUN_NAME=rna_ss
DATASET_NAME=$2 ## set the named of the dataset
CKPT_SAVE_DIR=logs/${RUN_NAME}/${DATASET_NAME} 

unset SLURM_NTASKS
if [ $MODE == "train" ]; then
	mgen fit --config rna_ss_prediction.yaml \
			--data.path /work/c30636/rna_reasoning/rna_ss_data/ \
			--data.dataset ${DATASET_NAME} \
			--trainer.default_root_dir ${CKPT_SAVE_DIR} \
			--trainer.devices 0,1 \
			--trainer.callbacks.ft_schedule_path ft_schedules/stage_2_fusion_gate.yaml \
			# --ckpt_path /work/c30636/rna_reasoning/rna_ss_reasoning/experiments/AIDO.RNA/rna_secondary_structure_prediction/logs/rna_ss/ensemble/lightning_logs/version_4/checkpoints/ss-epoch=9-valF1_no_relax=0.893.ckpt
else
	CKPT_PATH=$3 ## set the path to the checkpoint file (example shown in the commented line above)
	mgen test --config rna_ss_prediction.yaml \
			--data.path /work/c30636/rna_reasoning/rna_ss_data/ \
			--data.dataset ${DATASET_NAME} \
			--trainer.default_root_dir ${CKPT_SAVE_DIR} \
			--trainer.callbacks.ft_schedule_path ft_schedules/layers_0_32.yaml \
			--ckpt_path ${CKPT_PATH} \
			--trainer.devices 0,
fi