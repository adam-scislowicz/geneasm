# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.8.16 ('base')
#     language: python
#     name: python3
# ---

# %%
# !pip3 install biopython scikit-bio torchinfo hnswlib

# %%
import os
import json

from Bio import Entrez, SeqIO
from Bio.Seq import Seq

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchinfo import summary
from tqdm.notebook import tqdm

from skbio.alignment import StripedSmithWaterman

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# %%
cfg = dict()
cfg['kmer_size'] = 64
cfg['embedding_dim'] = 512
cfg['batch_size'] = 128
cfg['epochs'] = 10
cfg['report_every_n_batches'] = 200
cfg['dense_layer_units'] = 8192
cfg['learning_rate'] = 0.0000075
cfg['num_inputs'] = 100000
cfg['max_snps'] = 10
cfg['alphabet_size'] = 16
cfg['conv1d_filters'] = [32, 64, 128]
cfg['conv1d_kernels'] = [3, 3, 3]
cfg['fc_output'] = 8192
cfg['use_pytorch_amp'] = True
cfg['force_retrain_model'] = True
cfg['parameter_dtype'] = 'float16'
cfg['model_bin'] = 'data/model.bin'
cfg['loss_mode'] = 'siamese'
cfg['triplet_loss_margin'] = 0.50
cfg['force_recalculate_inputs'] = False
cfg['input_x_path'] = 'data/input_x'
cfg['input_y_path'] = 'data/input_y'
cfg['train_loader_path'] = '/opt/ml/data/test/train.dat'
cfg['test_loader_path'] = '/opt/ml/data/test/test.dat'
cfg['sagemaker_hyperparameter_path'] = '/opt/ml/input/config/hyperparameters.json'

# %%
Entrez.email = 'anonymous@outlook.com'
DATA_DIR='./data'

# fetch samples from Entrez, the US federal database
# TODO: later these will be used to cross-validate as additional testing rounds.
# This was done historically and performance was very similar to with the random data
# supporting the training methods application to non-random input.

def EntrezFetch(db, accession, type='fasta', mode='text'):
    cached_path = DATA_DIR+'/'+db+'-'+accession
    handle = Entrez.efetch(db=db, id=accession, rettype=type, retmode=mode)
    os.makedirs(os.path.dirname(cached_path), exist_ok=True)
    with open(cached_path, "w") as f:
        f.write(handle.read())
    return cached_path

def load_entrez_samples():
    print("Downloading test samples for later validation...")
    datapath = EntrezFetch('nucleotide', 'EU054331')
    print(datapath)

    records = SeqIO.parse(datapath, "fasta")

    print("Done processing test samples.")

#for rec in records:
#    seq = Seq(str(rec.seq))
#    print(seq)


# %%
if torch.cuda.is_available(): 
 dev = "cuda:0" 
else: 
 dev = "cpu"
 if cfg['parameter_dtype'] == "float16":
   cfg['parameter_dtype'] = "bfloat16"
device = torch.device(dev)
print(dev)


# %%
class SWAEmb(nn.Module):
    def __init__(self):
        super(SWAEmb, self).__init__()
    
        self.linear_fc_input = (cfg['kmer_size'] - cfg['conv1d_kernels'][0] + 1) // 2
        self.linear_fc_input = \
            (self.linear_fc_input - cfg['conv1d_kernels'][1] + 1) // 2
        self.linear_fc_input = \
            ((self.linear_fc_input - cfg['conv1d_kernels'][2] + 1) * cfg['conv1d_filters'][2]) // 2
        
        self.net = nn.Sequential(
            nn.Conv1d(cfg['alphabet_size'], cfg['conv1d_filters'][0], cfg['conv1d_kernels'][0], stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cfg['conv1d_filters'][0], cfg['conv1d_filters'][1], cfg['conv1d_kernels'][1], stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cfg['conv1d_filters'][1], cfg['conv1d_filters'][2], cfg['conv1d_kernels'][2], stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(in_features=self.linear_fc_input, out_features=cfg['fc_output']),
            nn.ReLU(),
            nn.Linear(in_features=cfg['fc_output'], out_features=cfg['embedding_dim'])
        )

    def forward_once(self, x):
        x = self.net(x)
        return x

    def forward(self, X):
        A = X[:,:,:,0]
        B = X[:,:,:,1]
        oA = self.forward_once(A)
        oB = self.forward_once(B)

        return oA, oB

class SiameseLoss(nn.Module):
    def __init__(self, debug_metrics=False):
        super(SiameseLoss, self).__init__()

        self.debug_metrics = debug_metrics
        if debug_metrics:
            self.absim_samples = np.empty(shape=(0), dtype=cfg['parameter_dtype'])
            self.align_loss_samples = np.empty(shape=(0), dtype=cfg['parameter_dtype'])
            
    # TODO: properly utilise stop gradient to avoid the autodifferentation agent from
    #       applying backprop to the wrong variables.
    def forward(self, distance, ABscore):
        loss_contrastive = torch.mean((1 - ABscore) * torch.pow(distance, 2) +
                                      (ABscore) * torch.pow(torch.clamp(0.25 - distance, min=0.0), 2))
        
        return loss_contrastive
    
class TripletLoss(nn.Module):
    def __init__(self, debug_metrics=False, margin=cfg['triplet_loss_margin']):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
        self.debug_metrics = debug_metrics
        if debug_metrics:
            self.align_loss_samples = np.empty(shape=(0), dtype=cfg['parameter_dtype'])
            self.triplet_loss_samples = np.empty(shape=(0), dtype=cfg['parameter_dtype'])
            self.triplet_loss_margin_samples = np.empty(shape=(0), dtype=cfg['parameter_dtype'])
            self.sw_delta = np.empty(shape=(0), dtype=cfg['parameter_dtype'])


    # This loss function is nonsense. I queried ChatGPT and modified it. This after porting
    # from tensorflow. Revisit later if interested.
    def forward(self, anchor, positive, negative, APscore, ANscore):
        ap_sim = F.cosine_similarity(anchor, positive, dim=1)
        an_sim = F.cosine_similarity(anchor, negative, dim=1)
        align_loss = F.mse_loss(ap_sim, APscore) + F.mse_loss(an_sim, ANscore)
        triplet_loss = torch.sum(torch.max((1.0 - ap_sim) - (1.0 - an_sim) + self.margin), 0)
        tp = ap_sim.detach().cpu()
        tn = an_sim.detach().cpu()
        
        if self.debug_metrics:
            self.align_loss_samples = np.append(self.align_loss_samples, align_loss.detach().cpu().numpy())
            for i in range(len(APscore)):
                self.sw_delta = np.append(self.sw_delta, APscore[i].cpu()-ANscore[i].cpu())
            for i in range(len(APscore)):
                self.triplet_loss_margin_samples = np.append(self.triplet_loss_margin_samples, [torch.sum(torch.max((1.0 - tp[i]) - (1.0 - tn[i]) + self.margin), 0).numpy(), (1-tp[i]) - (1-tn[i])], axis=0)
        loss = align_loss + triplet_loss
        loss = loss.to(torch.device(dev))
        return loss

# %%
alphabet = ['a', 'c', 'g', 't', 'r', 'y', 's', 'w', 'k', 'm', 'b', 'd', 'h', 'v', 'n', '-']
alphabet_to_onehoti = dict()
onehoti_to_alphabet = dict()
alphabet_len = len(alphabet)
assert(alphabet_len == cfg['alphabet_size'])

# populate map and reverse map
for i, c in enumerate(alphabet):
    alphabet_to_onehoti[c] = i
    onehoti_to_alphabet[i] = c
    
def char_to_onhot(c):
    oh = np.zeros((alphabet_len), dtype=cfg['parameter_dtype'])
    i = alphabet_to_onehoti[c]
    oh[i] = 1
    return oh
    
def seqstr_to_onehot(seqstr):
    size = len(seqstr)
    oh = np.zeros((size, alphabet_len), dtype=cfg['parameter_dtype'])
    for off, c in enumerate(seqstr):
        i = alphabet_to_onehoti[c]
        oh[off][i] = 1
    return oh

def onehot_to_seqstr(onehot):
    size = onehot.shape[0]
    seqstr = str()
    for pos in range(size):
        i = next((idx for idx, val in np.ndenumerate(onehot[pos,:]) if val==1))[0]
        seqstr += onehoti_to_alphabet[i]
    return seqstr

def gen_random_seq(seqlen, alphabet):
    seq = ""
    alphabet_len = len(alphabet)
    for pos in range(seqlen):
        cidx = np.random.choice(alphabet_len)
        seq += alphabet[cidx]
    return seq

# %%


def generate_triplet_input(num_triplets, phase_delta_min, phase_delta_max, max_snps,
                      ndelta_phase, ndelta_snps):
    assert(ndelta_phase != 0)
    assert(ndelta_snps != 0)
        
    m = 0
    triplets = np.empty(shape=(num_triplets, int(cfg['kmer_size']), alphabet_len, 3), dtype=cfg['parameter_dtype'])
    triplet_swas = np.empty(shape=(num_triplets, 2), dtype=cfg['parameter_dtype'])
    
    while m < num_triplets:
        
        # generate A(nchor)
        phase_gamut = (phase_delta_max - phase_delta_min)
        # TODO: Use weighted alphabet here as well
        SA = gen_random_seq(int(cfg['kmer_size'])+(phase_gamut * 2), alphabet=['g', 'c', 'a', 't'])
        A = SA[phase_gamut:int(cfg['kmer_size'])+phase_gamut]
    
        # generate permutation spec for P(ositive)
        phase_delta = np.random.choice(phase_delta_max - phase_delta_min) + phase_delta_min
        num_snps = np.random.choice(max_snps)
    
        # generate permutation spec for N(egative)
        while True:
            nphase_delta = np.random.choice(ndelta_phase)
            if phase_delta < 0:
                nphase_delta *= -1
            nphase_delta = phase_delta + nphase_delta
            nnum_snps = np.random.choice(ndelta_snps) + num_snps
            
            # recalculate if N is no further from the anchor than P
            if (nphase_delta == phase_delta) and (num_snps == nnum_snps):
                continue
            break
    
        P = SA[phase_gamut+phase_delta:int(cfg['kmer_size'])+phase_gamut+phase_delta]
        offs = set()
        for i in range(num_snps):
            while True:
                off = np.random.choice(cfg['kmer_size'])
                idx = np.random.choice(len(alphabet))
                if P[off] == alphabet[idx]:
                    continue
                if off in offs:
                    continue
                offs.add(off)
                l=list(P)
                l[off] = alphabet[idx]
                P=''.join(l)
                break
            
        N = SA[phase_gamut+nphase_delta:int(cfg['kmer_size'])+phase_gamut+nphase_delta]
        offs = set()
        for i in range(nnum_snps):
            while True:
                off = np.random.choice(cfg['kmer_size'])
                idx = np.random.choice(len(alphabet))
                if N[off] == alphabet[idx]:
                    continue
                if off in offs:
                    continue
                offs.add(off)
                l=list(N)
                l[off] = alphabet[idx]
                N=''.join(l)
                break
    
        #print("A:", A)
        #print("P:", phase_delta, P)
        #print("N:", nphase_delta, N)
        triplets[m,:,:,0] = seqstr_to_onehot(A)
        triplets[m,:,:,1] = seqstr_to_onehot(P)
        triplets[m,:,:,2] = seqstr_to_onehot(N)
        ssw = StripedSmithWaterman(A, score_size=2)
        
        # TODO: The SWAs (optimal alignment scores) are packed to close to each other
        # need to set SW params to acheive higher score gamut without distortion
        # then apply affine transforation (scale) to fill paramater datatype repr.
        triplet_swas[m,0] = torch.tensor(ssw(P)['optimal_alignment_score'])
        triplet_swas[m,1] = torch.tensor(ssw(N)['optimal_alignment_score'])
        if (triplet_swas[m,0] == triplet_swas[m,1]):
            continue
        
        m += 1
        if (m % 1000) == 0:
            print("Triplet {}".format(m))
    
    print("Print auto-scaling triplet_swas...")
    minSwa = np.min(triplet_swas[:,0])
    tmp = np.min(triplet_swas[:,1])
    if tmp < minSwa:
        minSwa = tmp
    maxSwa = np.max(triplet_swas[:,0])
    tmp = np.max(triplet_swas[:,1])
    if tmp > maxSwa:
        maxSwa = tmp
    swaDelta = maxSwa - minSwa
    triplet_swas[:,0] -= minSwa
    triplet_swas[:,1] -= minSwa
    triplet_swas[:,0] /= swaDelta
    triplet_swas[:,1] /= swaDelta
    print("Done auto-scaling triplet swas.")
    
    return triplets, triplet_swas
# %%
def generate_siamese_input(num_inputs, phase_delta_min, phase_delta_max, max_snps,
                           ndelta_phase, ndelta_snps):
    assert(ndelta_phase != 0)
    assert(ndelta_snps != 0)
        
    m = 0
    input_x = np.empty(shape=(num_inputs, int(cfg['kmer_size']), alphabet_len, 3), dtype=cfg['parameter_dtype'])
    input_y = np.empty(shape=(num_inputs, 2), dtype=cfg['parameter_dtype'])
    
    while m < num_inputs:
        
        phase_gamut = (phase_delta_max - phase_delta_min)
        
        # generate seqA
        SA = gen_random_seq(int(cfg['kmer_size'])+(phase_gamut * 2), alphabet=['g', 'c', 'a', 't'])
        A = SA[phase_gamut:int(cfg['kmer_size'])+phase_gamut]
        
        # generate seqB
        SB = gen_random_seq(int(cfg['kmer_size'])+(phase_gamut * 2), alphabet=['g', 'c', 'a', 't'])
        B = SB[phase_gamut:int(cfg['kmer_size'])+phase_gamut]
    
        #print("A:", A)
        #print("B:", B)
        input_x[m,:,:,0] = seqstr_to_onehot(A)
        input_x[m,:,:,1] = seqstr_to_onehot(B)
        ssw = StripedSmithWaterman(A, score_size=2)
        
        input_y[m,0] = torch.tensor(ssw(B)['optimal_alignment_score'])
        
        m += 1
        if (m % 1000) == 0:
            print("Pair {}".format(m))
    
    print("Auto-scaling smith waterman scores (expectation values) with respect to the input set...")
    minSwa = np.min(input_y[:,0])
    maxSwa = np.max(input_y[:,0])
    swaDelta = maxSwa - minSwa
    input_y[:,0] -= minSwa
    input_y[:,0] /= swaDelta
    print("Done auto-scaling expectation values.")
    
    return input_x, input_y


# %%
def init_inputs():
    if os.path.isfile(cfg['input_x_path']+'.npy') and not cfg['force_recalculate_inputs']:
        print("Loading input from file...")
        input_x = np.load(cfg['input_x_path']+'.npy')
        input_y = np.load(cfg['input_y_path']+'.npy')
        print("done loading input from file.")
    else:
        if cfg['loss_mode'] == 'siamese':
            input_x, input_y = generate_siamese_input(num_inputs=cfg['num_inputs'],
                                                    phase_delta_min=-int(cfg['kmer_size']),
                                                    phase_delta_max=int(cfg['kmer_size']),
                                                    max_snps=cfg['max_snps'],
                                                    ndelta_phase=2, ndelta_snps=2)
        else:
            input_x, input_y = generate_triplet_input(num_triplets=cfg['num_inputs'],
                                                    phase_delta_min=-int(cfg['kmer_size']),
                                                    phase_delta_max=int(cfg['kmer_size']),
                                                    max_snps=cfg['max_snps'],
                                                    ndelta_phase=2, ndelta_snps=2)
        print("Saving input to file...")
        np.save(cfg['input_x_path'], input_x)
        np.save(cfg['input_y_path'], input_y)
        print("done saving input to file.")
    
    print('done.')
    return input_x, input_y


# %%
# Map-type dataset
class CustomInputDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        self.labels = labels
        self.data = data
        if (dev != 'cpu'):
            self.data = torch.from_numpy(self.data).to(torch.device(dev))
            self.labels = torch.from_numpy(self.labels).to(torch.device(dev))

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):
        return self.data[index], self.labels[index]

def init_loaders(input_x, input_y):
      # separate inputs into train and test sets

      train_loader = None
      test_loader = None

      if os.path.isfile(cfg['train_loader_path']):
            train_loader = torch.load(cfg['train_loader_path'])
      if os.path.isfile(cfg['test_loader_path']):
            test_loader = torch.load(cfg['test_loader_path'])

      train_set_size = int(len(input_x) * 0.8)
      test_set_size = len(input_x) - train_set_size

      if (input_x.shape[1] == 64):
            input_x = np.transpose(input_x, (0,2,1,3)) # modify order for pytorch input

      if train_loader == None:
            train = CustomInputDataset(input_x[0:train_set_size], input_y[0:train_set_size])
            train_loader = torch.utils.data.DataLoader(train, batch_size=cfg['batch_size'], shuffle=True)
            torch.save(test_loader, cfg['train_loader_path'])
            print("Train set size:", train_set_size)

      if test_loader == None:
            test = CustomInputDataset(input_x[-test_set_size:], input_y[-test_set_size:])
            test_loader = torch.utils.data.DataLoader(test, batch_size=cfg['batch_size'], shuffle=False)
            torch.save(train_loader, cfg['test_loader_path'])
            print("Test set size:", test_set_size)
      
      print('done.')

      return train_loader, test_loader


# %%
def train(train_loader):
    # Training
    model = SWAEmb()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['use_pytorch_amp'])

    requireTraining=True
    if os.path.isfile(cfg['model_bin']) and not cfg['force_retrain_model']:
        print("Loading saved model...")
        model.load_state_dict(torch.load(cfg['model_bin']))
        print("done loading saved model.")
        requireTraining = False
    
    if (dev != 'cpu'):
        model.to(torch.device(dev))

    if cfg['loss_mode'] == 'siamese':
        criterion = SiameseLoss(debug_metrics=False)
    else:
        criterion = TripletLoss(debug_metrics=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate']) 

    if requireTraining:
        for epoch in range(cfg['epochs']):
            batchNum = 0
            totalLoss = 0
            total = 0
            print("Training EPOCH {} of {}:".format(epoch, cfg['epochs']))
            for X, E in train_loader:
                autocast_devtype = 'cpu'
                if (dev != 'cpu'):
                    autocast_devtype = 'cuda'
                with torch.autocast(device_type=autocast_devtype, dtype=getattr(torch, cfg['parameter_dtype']), enabled=cfg['use_pytorch_amp']):
                    if cfg['loss_mode'] == 'siamese':
                        yA, yB = model.forward(X)
                        loss = criterion(F.cosine_similarity(yA, yB), E[:,0])
                    else:
                        yA, yP, yN = model.forward(X)
                        loss = criterion(yA, yP, yN, E[:,0], E[:,1])
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
                optimizer.zero_grad()
            
                total += yA.shape[0]
                totalLoss += loss
                batchNum += 1
    
                if (batchNum % cfg['report_every_n_batches'] == 0) and (batchNum != 0):
                    print("    Training batch #{}: samples={}, totalLoss={}".format(batchNum, total, totalLoss))
                    totalLoss = 0

        print("Training complete. saving model...")
        torch.save(model.state_dict(), cfg['model_bin'])
        print("done saving model.")
        return model, criterion


# %%
def stats_and_plots():
    print("Min/Max/Avg Alignment Loss: {}/{}/{}".format(
        np.min(criterion.align_loss_samples),
        np.max(criterion.align_loss_samples),
        np.sum(criterion.align_loss_samples)/len(criterion.align_loss_samples)))

    if cfg['loss_mode'] != 'siamese':
        print("Min/Max/Avg SWDelta: {}/{}/{}".format(
            np.min(criterion.sw_delta),
            np.max(criterion.sw_delta),
            np.sum(criterion.sw_delta)/len(criterion.sw_delta)))

        if (criterion.triplet_loss_margin_samples.shape[0] > 800000):
            criterion.triplet_loss_margin_samples = criterion.triplet_loss_margin_samples.reshape(-1, 2)
            criterion.triplet_loss_margin_samples = np.transpose(criterion.triplet_loss_margin_samples, (1,0))

        print("Min/Max/Avg Raw Pcosdist-Ncosdist: {}/{}/{}".format(
            np.min(criterion.triplet_loss_margin_samples[1]),
            np.max(criterion.triplet_loss_margin_samples[1]),
            np.sum(criterion.triplet_loss_margin_samples[1])/len(criterion.triplet_loss_margin_samples[1])))

        print("Min/Max/Avg ReLU(Pcosdist-Ncosdist+margin): {}/{}/{}".format(
            np.min(criterion.triplet_loss_margin_samples[0]),
            np.max(criterion.triplet_loss_margin_samples[0]),
            np.sum(criterion.triplet_loss_margin_samples[0])/len(criterion.triplet_loss_margin_samples[0])))

    if cfg['loss_mode'] == 'siamese':
        fig = plt.figure(figsize=(10,10))

        ax1 = fig.add_subplot(4, 1, 1)
        plt.subplots_adjust(hspace=0.8)

        ax1.plot(criterion.align_loss_samples)
        ax1.set_title("Align Loss Samples")
        #ax1.set_ylim(0.04, 0.16)

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.hist(criterion.align_loss_samples, bins=200)
        ax2.set_title("Align Loss Histogram")
        ax2.set_yscale('log')
 
        ax3 = fig.add_subplot(4, 1, 3)
        ax3.plot(criterion.absim_samples)
        ax3.set_title("AB Similarity Samples")
        #ax3.set_ylim(0.04, 0.16)

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.hist(criterion.absim_samples, bins=200)
        ax4.set_title("AB Similarity Histogram")
        ax4.set_yscale('log')
    
    else:
        fig = plt.figure(figsize=(10,10))

        ax1 = fig.add_subplot(5, 1, 1)
        plt.subplots_adjust(hspace=0.8)

        ax1.plot(criterion.align_loss_samples)
        ax1.set_title("Align Loss Samples")
        ax1.set_ylim(0.04, 0.16)

        ax2 = fig.add_subplot(5, 1, 2)
        ax2.hist(criterion.align_loss_samples, bins=200)
        ax2.set_title("Align Loss Histogram")
        ax2.set_yscale('log')
    
        ax5 = fig.add_subplot(5, 1, 3)
        ax5.plot(criterion.triplet_loss_margin_samples[1])
        ax5.set_title("Raw (Pcosdist - Ncosdist)")

        ax6 = fig.add_subplot(5, 1, 4)
        ax6.plot(criterion.sw_delta)
        ax6.set_title("Raw SWDelta")

        ax7 = fig.add_subplot(5, 1, 5)
        ax7.plot(criterion.triplet_loss_margin_samples[0])
        ax7.set_title("ReLU(Triplet Loss Samples w/ Margin)")
        plt.show()


# %%

def normalized_concordance(sA, sB, swa):
    tspan = swa['target_end_optimal'] - swa['target_begin']
    tval = sB[swa['target_begin']:swa['target_end_optimal']+1]
    qspan = swa['query_end'] - swa['query_begin']
    qval = sA[swa['query_begin']:swa['query_end']+1]
    concordance = sum(a==b for a, b in zip(tval, qval))
    return concordance / cfg['kmer_size']

def get_three_uniqe_indexes(max_index):
    vals=np.empty((0), dtype=int)
    while len(vals) < 3:
        val = np.random.choice(max_index-1)
        if val in vals:
            continue
        vals = np.append(vals, val)
    return vals

# Testing
def test(model, criterion, test_loader):
    batchNum = 0
    totalLoss = 0
    total = 0
    match = 0
    mismatch = 0
    discernment_error = np.empty((0))
    mismatch_error = np.empty((0))
    mismatching_samples=np.empty((0))
    for X, E in tqdm(test_loader):

        with torch.autocast(device_type=autocast_devtype, dtype=getattr(torch, cfg['parameter_dtype']), enabled=cfg['use_pytorch_amp']):
            if cfg['loss_mode'] == 'siamese':
                yA, yB = model.forward(X)
                loss = criterion(F.cosine_similarity(yA, yB), E[:,0])
            else:
                yA, yP, yN = model.forward(X)
                loss = criterion(yA, yP, yN, E[:,0], E[:,1])
        
        total += yA.shape[0]
        totalLoss += loss
    
        for i, iX in enumerate(X):
            if False and cfg['loss_mode'] != 'siamese': # for debugging purposes.
                iA = torch.transpose(iX[:, :, 0], 0, 1)
                iP = torch.transpose(iX[:, :, 1], 0, 1)
                iN = torch.transpose(iX[:, :, 2], 0, 1)
        
                if (dev != 'cpu'):
                    iA = iA.cpu()
                    iP = iP.cpu()
                    iN = iN.cpu()
            
                sA = onehot_to_seqstr(iA)
                sP = onehot_to_seqstr(iP)
                sN = onehot_to_seqstr(iN)
                #print("sA: {}".format(sA))
                #print("sP: {}".format(sP))
                #print("sN: {}".format(sN))
                sswA = StripedSmithWaterman(sA, score_size=2)
                swaAP = sswA(sP)
                swaAN = sswA(sN)
        
                nconcordAP = normalized_concordance(sA, sP, swaAP)
                nconcordAN = normalized_concordance(sA, sN, swaAN)
                #print("ssw(A,P): {}, ssw(A,N): {}".format(
                #    swaAP['optimal_alignment_score'],
                #    swaAN['optimal_alignment_score']
                #))
            

            if cfg['loss_mode'] == 'siamese':
                embestAB = F.cosine_similarity(yA[i].unsqueeze(0), yB[i].unsqueeze(0)).item()
            else:
                embestAP = F.cosine_similarity(yA[i].unsqueeze(0), yP[i].unsqueeze(0)).item()
                embestAN = F.cosine_similarity(yA[i].unsqueeze(0), yN[i].unsqueeze(0)).item()
   
                #print(embestAP-embestAN)
                if (embestAN > embestAP):
                    discernment_error = np.append(discernment_error, embestAN-embestAP)
    
        while True:
            a1i, a2i, a3i = get_three_uniqe_indexes(E.shape[0])
            i1 = torch.transpose(X[a1i, :, :, 0], 0, 1)
            i2 = torch.transpose(X[a2i, :, :, 0], 0, 1)
            i3 = torch.transpose(X[a3i, :, :, 0], 0, 1)
            y1 = yA[a1i].unsqueeze(0)
            y2 = yA[a2i].unsqueeze(0)
            y3 = yA[a3i].unsqueeze(0)
        
            if (dev != 'cpu'):
                i1 = i1.cpu()
                i2 = i2.cpu()
                i3 = i3.cpu()
           
            s1 = onehot_to_seqstr(i1)
            s2 = onehot_to_seqstr(i2)
            s3 = onehot_to_seqstr(i3)
            ssw1 = StripedSmithWaterman(s1, score_size=2)
            swa12 = ssw1(s2)
            swa13 = ssw1(s3)
        
            embest12 = F.cosine_similarity(y1, y2).item()
            embest13 = F.cosine_similarity(y1, y3).item()
        
            if swa12['optimal_alignment_score'] == swa13['optimal_alignment_score']:
                continue
            
            if swa12['optimal_alignment_score'] > swa13['optimal_alignment_score']:
                if embest12 > embest13:
                    match += 1
                else:
                    mismatch += 1
                    mismatch_error = np.append(mismatch_error, embest13-embest12)
                    mismatching_samples = np.append(mismatching_samples, [s1, s2, s3, embest12, embest13, swa12['optimal_alignment_score'], swa13['optimal_alignment_score']])
            else:
                if embest13 > embest12:
                    match += 1
                else:
                    mismatch += 1
                    mismatch_error = np.append(mismatch_error, embest12-embest13)
                    mismatching_samples = np.append(mismatching_samples, [s1, s2, s3, embest12, embest13, swa12['optimal_alignment_score'], swa13['optimal_alignment_score']])
            break
        
        batchNum = batchNum + 1
        if (batchNum % 100 == 0) and (batchNum != 0):
            print("Batch: {}...".format(batchNum))

    print("Total: {}, NormLoss: {}, Match: {} of {} ({})".format(
        total, (totalLoss/total), match, match+mismatch, match/(match+mismatch)))


# %%
def test_stats_and_plots():
    max_mismatch_error = np.max(mismatch_error)

    if cfg['loss_mode'] == 'siamese':
        fig, axs = plt.subplots(1, sharex=False)
        plt.subplots_adjust(hspace=0.5)
        axs.hist(mismatch_error, bins=40, range=[0.0, max_mismatch_error])
        axs.set_title("SWAEmb Type-I Estimation Error max={}".format(max_mismatch_error))
    else:
        max_discernment_error = np.max(discernment_error)
        fig, axs = plt.subplots(2, sharex=False)
        plt.subplots_adjust(hspace=0.5)
        axs[0].hist(discernment_error, bins=40, range=[0.0, max_discernment_error])
        axs[0].set_title("Discernment Error max={}".format(max_discernment_error))
        axs[1].hist(mismatch_error, bins=40, range=[0.0, max_mismatch_error])
        axs[1].set_title("SWAEmb Type-I Estimation Error max={}".format(max_mismatch_error))
# %%
def sandbox():
    s1 = 'ttgaatccctacgatgatcgcaaatgtagcacccccaagtcgctggccagcagagggttgatgt'
    s2 = 'ctctggaggcaagatggttctcgcttggtatgacacttacccaatgttgagctaagcccacagg'
    s3 = 'catgcctggccagttggtgggacagcgtattattaatgaaagatgggaaggacgtgccgagtca'
    ssw1 = StripedSmithWaterman(s1, score_size=2)
    ssw2 = StripedSmithWaterman(s2, score_size=2)
    ssw3 = StripedSmithWaterman(s3, score_size=2)
    print(ssw1(s2)['optimal_alignment_score'])
    print(ssw2(s1)['optimal_alignment_score'])
    print(ssw1(s3)['optimal_alignment_score'])
    print(ssw3(s1)['optimal_alignment_score'])

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(mismatching_samples)
    print(df)


# %%
if __name__ == "__main__":
    sagemaker_training_mode = False
    # Are we being run as a Sagemaker training job
    if os.path.isfile(cfg['sagemaker_hyperparameter_path']):
        sagemaker_training_mode = True
        with open(cfg['sagemaker_hyperparameter_path'], "r") as read_file:
            hyperparams_dict = json.load(read_file)
        print(hyperparams_dict)

    input_x, input_y = init_inputs()
    train_loader, test_loader = init_loaders(input_x, input_y)

    model, criterion = train(train_loader)
    test(model, criterion, test_loader)
