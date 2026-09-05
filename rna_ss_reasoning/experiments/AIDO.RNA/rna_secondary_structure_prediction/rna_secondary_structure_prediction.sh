# Sazan Mahbub
MODE=$1 ## set it to "train" for finetuning the RNA-FM for RNA secondary structure prediction

RUN_NAME=rna_ss
DATASET_NAME=$2 ## set the named of the dataset
CKPT_SAVE_DIR=logs/${RUN_NAME}/${DATASET_NAME} 

unset SLURM_NTASKS

CKPT_PATH=$3 ## set the path to the checkpoint file (example shown in the commented line above)
mgen test --config rna_ss_prediction.yaml \
		--data.path your_dataset_path \
		--data.dataset ${DATASET_NAME} \
		--trainer.default_root_dir ${CKPT_SAVE_DIR} \
		--trainer.callbacks.ft_schedule_path ft_schedules/layers_0_32.yaml \
		--ckpt_path ${CKPT_PATH} \
		--trainer.devices 0,