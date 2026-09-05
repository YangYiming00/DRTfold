WORKING_PATH= # you should set this environment variable to the path of DRTfold
PROTENIX_PATH=$WORKING_PATH/Protenix # you should set this environment variable to the path of Protenix
DRFOLD2_PATH=$WORKING_PATH/DRfold2 # you should set this environment variable to the path of DRfold2
REASONING_PATH=$WORKING_PATH/rna_ss_reasoning/experiments/AIDO.RNA/rna_secondary_structure_prediction
FASTA_FILE=$WORKING_PATH/input.fasta
MODEL_PATH= # you should set this environment variable to the path of the model checkpoint file

export FASTA_FILE=$FASTA_FILE
export PROTENIX_PATH=$PROTENIX_PATH
export DRFOLD2_PATH=$DRFOLD2_PATH
export SAVE_PATH=$WORKING_PATH

echo "FASTA: $FASTA_FILE"
cd $PROTENIX_PATH

echo "========================="
echo "Creating input json file for Protenix..."

python input/get_new_json.py

echo "Generating Protenix embeddings..."
LAYERNORM_TYPE=torch python runner/inference.py \
    --sample_diffusion.N_sample 1 \
    --model_name protenix_mini_default_v0.5.0 \
    --seeds 42 \
    --dump_dir "$PROTENIX_PATH/output" \
    --input_json_path "$PROTENIX_PATH/input/input.json"

cd $DRFOLD2_PATH

echo "Generating DRfold2 embeddings..."

python cfg_95/test_modeldir.py \
    cuda \
    "$FASTA_FILE" \
    "$DRFOLD2_PATH/embeddings" \
    "$DRFOLD2_PATH/model_hub/cfg_95"

cd $REASONING_PATH
bash rna_secondary_structure_prediction.sh test $FASTA_FILE $MODEL_PATH

echo "All done!"