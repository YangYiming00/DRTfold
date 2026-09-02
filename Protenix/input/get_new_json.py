import json
import os
fasta_file = os.environ["FASTA_FILE"]
PROTENIX_PATH = os.environ["PROTENIX_PATH"]

out_path = os.path.join(PROTENIX_PATH, "input", "input.json")

def read_fasta(file_path):
    records = []
    with open(file_path, "r") as f:
        seq_id = None
        seq_lines = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(seq_lines)))
                seq_id = line[1:].split()[0]  # 取 header 作为 id
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id is not None:
            records.append((seq_id, "".join(seq_lines)))

    return records

all_data = []
valid_bases = {"A", "U", "C", "G"}

records = read_fasta(fasta_file)

for rna_id, seq in records:
    seq = seq.upper()           
    seq = seq.replace("T", "U") # T -> U
    seq = "".join([b if b in valid_bases else "N" for b in seq])
    entry = {
        "sequences": [
            {
                "rnaSequence": {
                    "sequence": seq,
                    "count": 1
                }
            }
        ],
        "name": rna_id
    }
    all_data.append(entry)

print(f"Total sequences: {len(all_data)}")

with open(out_path, "w") as f:
    json.dump(all_data, f, indent=2)

print(f"Saved {len(all_data)} samples to {out_path}")