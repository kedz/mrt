from plum2.loss import Loss
from plum.types import Variable
import torch
from torch.nn.functional import kl_div


class PointerCrossEntropy(Loss):
    
    def __init__(self, logits_field="target_logits", labels_field="target", 
                 padding_index=-1, label_smoothing=0.0):
        super(PointerCrossEntropy, self).__init__()
        self.logits_field = logits_field
        self.labels_field = labels_field
        self.padding_index = padding_index
        self.label_smoothing = label_smoothing
        self._total_loss = 0
        self._num_items = 0

    def scalar_result(self):
        if self._num_items == 0:
            return 0
        else:
            return self._total_loss / self._num_items

    def reset(self):
        self._total_loss = 0
        self._num_items = 0

    def __call__(self, forward_state, batch):
        ignore_index = int(self.padding_index)
        logits = forward_state[self.logits_field]
        if isinstance(logits, Variable):
            logits = logits.permute_as_batch_sequence_features().data

        labels = batch[self.labels_field]
        label_lengths = labels.lengths

        if isinstance(labels, Variable):
            labels = labels.permute_as_batch_sequence_features()
            if labels.pad_value != None:
                labels = labels.apply_sequence_mask(pad_value=ignore_index)
            labels = labels.data

        with torch.no_grad():
            conf = 1. - float(self.label_smoothing)
            target_dist = logits.new(logits.size()).fill_(0)
            mask = logits.new().bool().new(logits.size()).fill_(0)
            #labels = labels.unsqueeze(1)
            for b in range(labels.size(0)):
                for t in range(label_lengths[b].item()):
                    smooth = (
                        float(self.label_smoothing) /
                        max(1, (labels[b,label_lengths[b].item()-1] - t).item())
                    )

                    for j in range(labels[b,label_lengths[b]-1].item() + 1):
                        target_dist[b,t,j] = smooth
                    target_dist[b, t, labels[b,t].item()].fill_(conf)
                    for j in range(t):
                        mask[b,t,labels[b,j].item()] = 1 #float('-inf')

                    for j in range(labels[b,label_lengths[b]-1].item() +1, logits.size(2)):
                        mask[b,t,j] = 1 #float('-inf')

            
        target_dist = target_dist.masked_fill(mask, 0)

        logits = logits.masked_fill(mask, float('-inf'))
        log_probs = logits.log_softmax(2)
        num_classes = logits.size(2)
        log_probs = log_probs.contiguous().view(-1, num_classes)
        target_dist = target_dist.contiguous().view(-1, num_classes)
       
#            labels = labels.contiguous().view(-1)
#            target_dist = target_dist.contiguous().view(-1, num_classes)


        tot_kld = kl_div(log_probs, target_dist, reduction='sum')
        num_items = label_lengths.sum().item()
        self._total_loss += tot_kld.item()
        self._num_items += num_items

        return tot_kld / num_items


        input()
        label_smoothing = float(self.label_smoothing) / (
            logits.size(2) - 1 - (1 if ignore_index > -1 else 0))
#        print(logits.size(2))
#        print(logits.size(2) - 1 - (1 if ignore_index > -1 else 0))
#        print(label_smoothing)
        target_dist = logits.data.clone().fill_(label_smoothing)

#        print(target_dist.size())
#        print(labels.size())
        conf = 1 - float(self.label_smoothing)
        target_dist.scatter_(2, labels.unsqueeze(2), conf)
#        print(target_dist)

#        self.true_dist = true_dist

        #target_dist.scatter_(
        #true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)


        if logits.dim() > 2:
            num_classes = logits.size(-1)
            logits = logits.contiguous().view(-1, num_classes)
            labels = labels.contiguous().view(-1)
            target_dist = target_dist.contiguous().view(-1, num_classes)

        if ignore_index > -1:
            target_dist[:, ignore_index] = 0
#            print(labels)
            mask = (labels == ignore_index).unsqueeze(1)

            
#            print(mask)
#            print(mask.size())

            target_dist = target_dist.masked_fill_(mask, 0.0)
#            print(target_dist[127,11])
#            print(target_dist[0,10])
#            print(labels[0,10])
#            print(label_lengths)
#

        if ignore_index is not None:
            num_items = labels.ne(ignore_index).long().sum().item()
        else: 
            num_items = labels.numel()
        
        tot_kld = kl_div(logits.log_softmax(1), target_dist, reduction='sum')

        self._total_loss += tot_kld.item()
        self._num_items += num_items

        return tot_kld / num_items

    def __kurrl__(self, forward_state, batch):
        if self.label_smoothing > 0.0:
            return self._forward_label_smoothing(forward_state, batch)

        ignore_index = -1 if self.padding_index == None \
            else int(self.padding_index)
        logits = forward_state[self.logits_field]
        if isinstance(logits, Variable):
            logits = logits.permute_as_batch_sequence_features().data

        labels = batch[self.labels_field]
        if isinstance(labels, Variable):
            labels = labels.permute_as_batch_sequence_features()
            if labels.pad_value != None:
                labels = labels.apply_sequence_mask(pad_value=ignore_index)
            labels = labels.data

        if logits.dim() > 2:
            num_classes = logits.size(-1)
            logits = logits.contiguous().view(-1, num_classes)
            labels = labels.contiguous().view(-1)

        if ignore_index is not None:
            num_items = labels.ne(ignore_index).long().sum().item()
        else: 
            num_items = labels.numel()

        total_xent = cross_entropy(
            logits, labels, ignore_index=ignore_index,
            reduction="sum")
        self._total_loss += total_xent.item()
        self._num_items += num_items

        return total_xent / num_items

    def compute(self):
        return {"cross_entropy": self.scalar_result()}

 
