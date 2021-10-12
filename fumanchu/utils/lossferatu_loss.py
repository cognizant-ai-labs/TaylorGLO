
import torch
import torch.nn as nn

class LossferatuLoss(torch.nn.Module):
    

    def __init__(self, loss_str, classification=True):
        super(LossferatuLoss, self).__init__()
        self.loss_str = loss_str
        self.classification = classification

    def forward(self, output, labels, training_progress):
        tf = torch

        # print("---Lossferatu OUTPUT:")
        # print(output.shape)
        # print("---Lossferatu LABELS:")
        # print(labels.shape)

        labels_flat = labels.view(-1)
        logits_flat = output
        if self.classification == True:
            preloss_softmax = torch.softmax(output, 1)
            onehot_labels = nn.functional.one_hot(labels_flat, num_classes=output.shape[1])
        
        # Loss function variables
        loss_x = onehot_labels if self.classification else labels
        loss_y = preloss_softmax if self.classification else output
        loss_t = torch.tensor(training_progress)
        loss_logits = logits_flat

        # Build and calculate loss
        sample_outputs_loss_vectors = eval(self.loss_str, globals(), locals()) if self.loss_str != None else loss_x * tf.log(loss_y)
        if self.classification == True:
            sample_losses = torch.mean(-torch.sum(sample_outputs_loss_vectors, 1))
        else:
            sample_losses = torch.mean(torch.sum(sample_outputs_loss_vectors, 1))
        mean_batch_loss = torch.mean(sample_losses)
        return mean_batch_loss
