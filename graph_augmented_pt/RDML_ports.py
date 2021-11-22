"""This file contains utilities for interfacing with and ported copies of code from
   https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/ (RDML).
"""

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

# Interface Utilities
class AdjMatrixLabels():
    """This class lets an Adjacency matrix simulate a static list of labels by overriding the ==, != and []
       operators. Instead of those actual operations, it computes:
         * [i] -> i (the identity of the node accessed,
         * == i -> [0 for all nodes with no link to i, 1 for all nodes with a link to i]
         * != i -> [1 for all nodes with no link to i, 0 for all nodes with a link to i]

       Both == and != error if the comparison is not to a valid node identifier.
    """

    @staticmethod
    def is_valid(adj_matrix):
        is_square = len(adj_matrix.shape) == 2 and adj_matrix.shape[0] == adj_matrix.shape[1]
        is_binary = (adj_matrix.bool().float() == adj_matrix.float()).all()

        return is_binary and is_square

    def __init__(self, adj_matrix):
        assert self.is_valid(adj_matrix), f"Adjacency matrix must be square & binary! Got {adj_matrix.shape}, {adj_matrix}"

        self.adj_matrix = adj_matrix.bool().cpu().detach().numpy()
        self.N = len(adj_matrix)

    def __getitem__(self, i): return i

    def __ne__(self, other):
        """This method actually returns a vector of what node `other` is not linked to"""
        assert other in range(self.N), f"{other} must be an index to a node (0 - {self.N-1}!"
        return ~self.adj_matrix[other]

    def __eq__(self, other):
        """This method actually returns a vector of what node `other` is linked to"""
        assert other in range(self.N), f"{other} must be an index to a node (0 - {self.N-1}!"
        return self.adj_matrix[other]

def produce_rdml_batchminer_input(points_encoded, context_encoded, gml_kwargs):
    """
    This function converts our graph-based metric-learning syntax into the syntax expected by Karsten's
    RDML Repository: https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
    """
    # Our final kwargs will be a carbon copy of what we're given, augmented with the extra kwargs, minus the
    # adjacency matrix, which we'll remove and convert below.
    batch = torch.cat((points_encoded['summary'], context_encoded), dim=0)
    labels = AdjMatrixLabels(gml_kwargs['adj_matrix'])

    return (batch, labels), {}

# RDML Ports:

## Batchminers

### Distance-weighted
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/batchminer/distance.py

class DistanceWeightedBatchMiner():
    def __init__(self, lower_cutoff, upper_cutoff):
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff

    def __call__(self, batch, labels, tar_labels=None, return_distances=False, distances=None):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs, dim = batch.shape

        if distances is None:
            distances = self.pdist(batch.detach()).clamp(min=self.lower_cutoff)
        sel_d = distances.shape[-1]

        positives, negatives = [],[]
        labels_visited       = []
        anchors              = []

        tar_labels = labels if tar_labels is None else tar_labels

        for i in range(bs):
            neg = tar_labels!=labels[i]; pos = tar_labels==labels[i]

            anchors.append(i)
            q_d_inv = self.inverse_sphere_distances(dim, bs, distances[i], tar_labels, labels[i])
            negatives.append(np.random.choice(sel_d,p=q_d_inv))

            if np.sum(pos)>0:
                #Sample positives randomly
                if np.sum(pos)>1: pos[i] = 0
                positives.append(np.random.choice(np.where(pos)[0]))
                #Sample negatives by distance

        sampled_triplets = [[a,p,n] for a,p,n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets


    def inverse_sphere_distances(self, dim, bs, anchor_to_all_dists, labels, anchor_label):
            dists  = anchor_to_all_dists

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dists) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dists.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse. It can also lead to
            # errors where there are no available negatives (for high samples_per_class cases).
            # q_d_inv[np.where(dists.detach().cpu().numpy()>self.upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()


    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.sqrt()

### Semi-hard Miner
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/batchminer/semihard.py
class SemiHardBatchMiner():
    def __init__(self, margin):
        self.margin       = margin

    def __call__(self, batch, labels, return_distances=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels!=l; pos = labels==l

            anchors.append(i)
            pos[i] = 0
            p      = np.random.choice(np.where(pos)[0])
            positives.append(p)

            #Find negatives that violate tripet constraint semi-negatives
            neg_mask = np.logical_and(neg,d>d[p])
            neg_mask = np.logical_and(neg_mask,d<self.margin+d[p])
            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]

        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets


    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()

### Soft-hard Miner
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/batchminer/softhard.py
class SoftHardBatchMiner():
    def __call__(self, batch, labels, return_distances=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels!=l; pos = labels==l

            if np.sum(pos)>1:
                anchors.append(i)
                #1 for batchelements with label l
                #0 for current anchor
                pos[i] = False

                #Find negatives that violate triplet constraint in a hard fashion
                neg_mask = np.logical_and(neg,d<d[np.where(pos)[0]].max())
                #Find positives that violate triplet constraint in a hard fashion
                pos_mask = np.logical_and(pos,d>d[np.where(neg)[0]].min())

                if pos_mask.sum()>0:
                    positives.append(np.random.choice(np.where(pos_mask)[0]))
                else:
                    positives.append(np.random.choice(np.where(pos)[0]))

                if neg_mask.sum()>0:
                    negatives.append(np.random.choice(np.where(neg_mask)[0]))
                else:
                    negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        if return_distances:
            return sampled_triplets, distances
        else:
            return sampled_triplets



    def pdist(self, A):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = 0).sqrt()

### n-Pair Miner
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/batchminer/npair.py
class NPairBatchMiner():
    def __call__(self, batch, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        anchors, positives, negatives = [],[],[]

        for i in range(len(batch)):
            anchor = i
            pos    = labels==labels[anchor]

            if np.sum(pos)>1:
                anchors.append(anchor)
                avail_positive = np.where(pos)[0]
                avail_positive = avail_positive[avail_positive!=anchor]
                positive       = np.random.choice(avail_positive)
                positives.append(positive)

        negatives = []
        for anchor,positive in zip(anchors, positives):
            neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive] and labels[i] != labels[anchor]]
            # neg_idxs = [i for i in range(len(batch)) if i not in [anchor, positive]]
            negative_set = np.arange(len(batch))[neg_idxs]
            negatives.append(negative_set)

        return anchors, positives, negatives

## Criteria

### Multisimilarity
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/multisimilarity.py

class MultisimilarityCriterion(torch.nn.Module):
    def __init__(self, pos_weight, neg_weight, margin, thresh):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.margin     = margin
        self.thresh     = thresh

    def forward(self, batch, labels, **kwargs):
        batch = F.normalize(batch, p=2, dim=1)
        similarity = batch.mm(batch.T)

        loss = []
        pos_terms, neg_terms = [], []
        for i in range(len(batch)):
            pos_idxs       = labels==labels[i]
            pos_idxs[i]    = 0
            neg_idxs       = labels!=labels[i]
            neg_idxs[i]    = 0

            if not np.sum(pos_idxs) or not np.sum(neg_idxs):
                continue
                
            anchor_pos_sim = similarity[i][pos_idxs]
            anchor_neg_sim = similarity[i][neg_idxs]

            ### This part doesn't really work, especially when you dont have a lot of positives in the batch...
            neg_idxs = (anchor_neg_sim + self.margin) > torch.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < torch.max(anchor_neg_sim)
            if not torch.sum(neg_idxs) or not torch.sum(pos_idxs):
                continue
            anchor_neg_sim = anchor_neg_sim[neg_idxs]
            anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1./self.pos_weight * torch.log(1+torch.sum(torch.exp(-self.pos_weight* (anchor_pos_sim - self.thresh))))
            neg_term = 1./self.neg_weight * torch.log(1+torch.sum(torch.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))
            loss.append(pos_term + neg_term)
            pos_terms.append(pos_term)
            neg_terms.append(neg_term)

        if loss == []:
            loss = torch.Tensor([0]).to(batch.device)
            pos_terms = torch.Tensor([0]).to(batch.device)
            neg_terms = torch.Tensor([0]).to(batch.device)
            loss.requires_grad = True
        else:
            loss = torch.mean(torch.stack(loss))
            pos_terms = torch.mean(torch.stack(pos_terms))
            neg_terms = torch.mean(torch.stack(neg_terms))
        return loss, {'pos': pos_terms, 'neg': neg_terms}

### N-Pair
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/npair.py

class NPairCriterion(torch.nn.Module):
    def __init__(self, l2_weight, npair_miner, value_clamping=False):
        super().__init__()
        self.value_clamping = value_clamping
        self.l2_weight = l2_weight
        self.batchminer = npair_miner

    def forward(self, batch, labels, **kwargs):
        anchors, positives, negatives = self.batchminer(batch, labels)

        ##
        loss  = 0
        if self.value_clamping:
            ### clamping/value reduction to avoid initial overflow for high embedding dimensions!
            batch = batch/4
        for anchor, positive, negative_set in zip(anchors, positives, negatives):
            a_embs, p_embs, n_embs = batch[anchor:anchor+1], batch[positive:positive+1], batch[negative_set]
            inner_sum = a_embs[:,None,:].bmm((n_embs - p_embs[:,None,:]).permute(0,2,1))
            inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
            loss  = loss + torch.mean(torch.log(torch.sum(torch.exp(inner_sum), dim=1) + 1))/len(anchors)
            loss  = loss + self.l2_weight*torch.mean(torch.norm(batch, p=2, dim=1))/len(anchors)

        return loss

### Triplet
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/triplet.py

class TripletCriterion(torch.nn.Module):
    def __init__(self, margin, batchminer):
        super().__init__()
        self.margin     = margin
        self.batchminer = batchminer

    def triplet_distance(self, anchor, pos, neg):
        return torch.nn.functional.relu((anchor-pos).pow(2).sum()-(anchor-neg).pow(2).sum()+self.margin)

    def forward(self, batch, labels, **kwargs):
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        sampled_triplets = self.batchminer(batch, labels)
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)

### Quadruplet
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/quadruplet.py

class QuadrupletCriterion(torch.nn.Module):
    def __init__(self, margin_alpha_1, margin_alpha_2, batchminer):
        super().__init__()
        self.batchminer = batchminer

        self.margin_alpha_1 = margin_alpha_1
        self.margin_alpha_2 = margin_alpha_2

    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu(torch.norm(anchor-positive, p=2, dim=-1)-torch.norm(anchor-negative, p=2, dim=-1)+self.margin_alpha_1)

    def quadruplet_distance(self, anchor, positive, negative, fourth_negative):
        return torch.nn.functional.relu(torch.norm(anchor-positive, p=2, dim=-1)-torch.norm(negative-fourth_negative, p=2, dim=-1)+self.margin_alpha_2)

    def forward(self, batch, labels, **kwargs):
        sampled_triplets    = self.batchminer(batch, labels)

        anchors   = np.array([triplet[0] for triplet in sampled_triplets]).reshape(-1,1)
        positives = np.array([triplet[1] for triplet in sampled_triplets]).reshape(-1,1)
        negatives = np.array([triplet[2] for triplet in sampled_triplets]).reshape(-1,1)

        fourth_negatives = negatives!=negatives.T
        fourth_negatives = [np.random.choice(np.arange(len(batch))[idxs]) for idxs in fourth_negatives]

        triplet_loss     = self.triplet_distance(batch[anchors,:],batch[positives,:],batch[negatives,:])
        quadruplet_loss  = self.quadruplet_distance(batch[anchors,:],batch[positives,:],batch[negatives,:],batch[fourth_negatives,:])

        return torch.mean(triplet_loss) + torch.mean(quadruplet_loss)

### Contrastive
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/contrastive.py
class ContrastiveCriterion(torch.nn.Module):
    def __init__(self, pos_margin, neg_margin, batchminer):
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.batchminer = batchminer

    def forward(self, batch, labels, **kwargs):
        sampled_triplets = self.batchminer(batch, labels)

        anchors   = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        pos_dists = torch.mean(F.relu(nn.PairwiseDistance(p=2)(batch[anchors,:], batch[positives,:]) -  self.pos_margin))
        neg_dists = torch.mean(F.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors,:], batch[negatives,:])))

        loss      = pos_dists + neg_dists

        return loss

### Adversarial Separation
### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/adversarial_separation.py
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class AdversarialSeparationCriterion(torch.nn.Module):
    def __init__(self, embed_dim, proj_dim, directions, weights):
        super().__init__()

        ####
        self.embed_dim  = embed_dim
        self.proj_dim   = proj_dim

        self.directions = directions
        self.weights    = weights

        #Projection network
        self.regressors = nn.ModuleDict()
        for direction in self.directions:
            self.regressors[direction] = torch.nn.Sequential(torch.nn.Linear(self.embed_dim, self.proj_dim), torch.nn.ReLU(), torch.nn.Linear(self.proj_dim, self.embed_dim)).to(torch.float).to(opt.device)

    def forward(self, feature_dict):
        #Apply gradient reversal on input embeddings.
        adj_feature_dict = {key:torch.nn.functional.normalize(grad_reverse(features),dim=-1) for key, features in feature_dict.items()}
        #Project one embedding to the space of the other (with normalization), then compute the correlation.
        sim_loss = 0
        for weight, direction in zip(self.weights, self.directions):
            source, target = direction.split('-')
            sim_loss += -1.*weight*torch.mean(torch.mean((adj_feature_dict[target]*torch.nn.functional.normalize(self.regressors[direction](adj_feature_dict[source]),dim=-1))**2,dim=-1))
        return sim_loss


### Gradient Reversal Layer
class GradRev(torch.autograd.Function):
    """
    Implements an autograd class to flip gradients during backward pass.
    """
    def forward(self, x):
        """
        Container which applies a simple identity function.
        Input:
            x: any torch tensor input.
        """
        return x.view_as(x)

    def backward(self, grad_output):
        """
        Container to reverse gradient signal during backward pass.
        Input:
            grad_output: any computed gradient.
        """
        return (grad_output * -1.)

### Gradient reverse function
def grad_reverse(x):
    """
    Applies gradient reversal on input.
    Input:
        x: any torch tensor input.
    """
    return GradRev()(x)
