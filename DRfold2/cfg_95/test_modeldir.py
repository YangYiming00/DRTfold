import random
random.seed(0)
import numpy as np
np.random.seed(0)
import os,sys,re,random
from numpy import select
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
expdir=os.path.dirname(os.path.abspath(__file__))



import torch.optim as opt
from torch.nn import functional as F
import data,util
import EvoMSA2XYZ,basic
import math
import pickle
Batch_size=3
Num_cycle=3
TEST_STEP=1000
VISION_STEP=50
device = sys.argv[1]


# expdir=os.path.dirname(os.path.abspath(__file__))
# expround=expdir.split('_')[-1]
# model_path=os.path.join(expdir,'others','models')
# if not os.path.isdir(model_path):
#     try:
#         os.makedirs(model_path)
#     except:
#         pass  
# testdir=os.path.join(expdir,'others','preds')
# if not os.path.isdir(testdir):
#     try:
#         os.makedirs(testdir)
#     except:
#         pass 



basenpy_standard= np.load( os.path.join(os.path.dirname(os.path.abspath(__file__)),'base.npy'  )  )
def data_collect(pdb_seq):
    aa_type = data.parse_seq(pdb_seq)
    # print(aa_type)
    base = data.Get_base(pdb_seq,basenpy_standard)
    seq_idx = np.arange(len(pdb_seq)) + 1

    msa=aa_type[None,:]
    msa=torch.from_numpy(msa).to(device)
    msa=torch.cat([msa,msa],0)
    msa=F.one_hot(msa.long(),6).float()

    base_x = torch.from_numpy(base).float().to(device)
    seq_idx = torch.from_numpy(seq_idx).long().to(device)
    return msa,base_x,seq_idx
    predxs,plddts = model.pred(msa,seq_idx,ss,base_x,sample_1['alpha_0'])



def classifier(infasta,out_prefix,model_dir):
    with torch.no_grad():
        # lines = open(infasta).readlines()[1:]
        # seqs = [aline.strip() for aline in lines]
        # print(lines)
        # print(seqs)
        # seq = ''.join(seqs)
        
        # seq_idx = np.genfromtxt(idxfile).astype(int)
        # seq_idx = torch.from_numpy(seq_idx).long().to(device)
        
        msa_dim=6+1
        m_dim,s_dim,z_dim = 64,64,64
        N_ensemble,N_cycle=3,8
        model=EvoMSA2XYZ.MSA2XYZ(msa_dim-1,msa_dim,N_ensemble,N_cycle,m_dim,s_dim,z_dim)
        model.to(device)
        model.eval()
        models = os.listdir(  model_dir   )
        models = [amodel for amodel in models if 'model' in amodel and 'opt' not in amodel]
        
        models.sort()

        names = []
        seqs = []

        with open(infasta, "r") as f:
            name = None
            seq = []
            
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if name is not None:
                        names.append(name)
                        seqs.append("".join(seq))
                    name = line[1:]  # 去掉 '>'
                    seq = []
                else:
                    seq.append(line)
            
            # 最后一条
            if name is not None:
                names.append(name)
                seqs.append("".join(seq))

        for name, seq in zip(names, seqs):
            save_path = f"{out_prefix}/{name}.pt"
            if os.path.exists(save_path):
                continue

            try:
                msa,base_x,seq_idx = data_collect(seq)

                for amodel in models:
                    #saved_model=os.path.join(expdir,'others','models',amodel)
                    saved_model=os.path.join(model_dir,amodel)
                    model.load_state_dict(torch.load(saved_model,map_location='cpu'),strict=True)
                    ret, z = model.pred(msa,seq_idx,None,base_x,np.array(list(seq)))
                    # print(z.shape)
                    torch.save(z.cpu(), save_path)
                    break

            except Exception as e:
                print(f"Error processing {name}: {e}")
                with open("/work/c30636/rna_reasoning/DRfold2/error.txt", "a") as f:
                    f.write(f"{name}\n")
                continue

            # util.outpdb(ret['coor'],seq_idx,seq,out_prefix+f'{amodel}.pdb')
            # #ret = {'plddt':ret['plddt']}
            # pickle.dump(ret,open(out_prefix+f'{amodel}.ret','wb'))

# python /lustre1/work/c30636/rna_reasoning/DRfold2/cfg_95/test_modeldir.py cuda /work/c30636/rna_reasoning/DRfold2/split_4.fasta /lustre1/work/c30636/rna_reasoning/DRfold2/embeddings /lustre1/work/c30636/rna_reasoning/DRfold2/model_hub/cfg_95

 
 
    

if __name__ == '__main__':
    infasta,out_prefix,model_dir = sys.argv[2],sys.argv[3],sys.argv[4]
    classifier(infasta,out_prefix,model_dir)