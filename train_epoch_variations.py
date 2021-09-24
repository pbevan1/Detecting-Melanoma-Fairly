import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torch.autograd import Variable
from globalbaz import args, DP, device
from tqdm import tqdm
from models import *


# Defining criterion with weighted loss based on bias to be unlearned
def criterion_func(df):
    lst = df['fitzpatrick'].value_counts().sort_index().tolist()
    lst2 = df['fitzpatrick'].value_counts().sort_index().tolist()  # placeholder

    sum_lst = sum(lst)
    sum_lst2 = sum(lst2)
    class_freq = []
    class_freq2 = []
    for i in lst:
        class_freq.append(i / sum_lst * 100)
    weights = torch.tensor(class_freq, dtype=torch.float32)
    for i in lst2:
        class_freq2.append(i / sum_lst2 * 100)
    weights2 = torch.tensor(class_freq2, dtype=torch.float32)

    weights = weights / weights.sum()
    weights2 = weights2 / weights2.sum()
    weights = 1.0 / weights
    weights2 = 1.0 / weights2
    weights = weights / weights.sum()
    weights2 = weights2 / weights2.sum()
    if args.debias_config != 'baseline':  # Only printing auxiliary weights head if using debiasing head
        print(f'weights_aux: {weights}')
        print(f'weights_aux_2: {weights2}')
    weights = weights.to(device)
    weights2 = weights2.to(device)
    # Note CrossEntropyLoss & BCEWithLogitsLoss includes the Softmax function so logits should be passed in (no softmax layer in model)
    criterion = nn.BCEWithLogitsLoss()  # nn.CrossEntropyLoss()
    criterion_aux = nn.CrossEntropyLoss(weight=weights)
    criterion_aux2 = nn.CrossEntropyLoss(weight=weights2)

    return criterion, criterion_aux, criterion_aux2


# Defining one training epoch for baseline model
def train_epoch_baseline(model_encoder, model_classifier, loader, optimizer, criterion):
    # Setting to train mode
    model_encoder.train()
    model_classifier.train()
    train_loss = []  # creating loss list
    bar = tqdm(loader)  # using tqdm to display progress bar
    for (data, target, _, _) in bar:
        optimizer.zero_grad()  # zeroing gradients
        data, target = data.to(device), target.to(device)  # sending data to GPU
        feat_out = model_encoder(data)  # creating feature representation using the encoder
        logits = model_classifier(feat_out)  # using the main classifier to get output logits
        target = target.unsqueeze(1).type_as(logits)  # unsqueezing to[batch_size,1] and same dtype as logits

        loss = criterion(logits, target)  # calculating loss using categorical crossentorpy

        loss.backward()  # backpropegating to calculate gradients

        optimizer.step()  # updating weights
        loss_np = loss.detach().cpu().numpy()  # sending loss to cpu
        train_loss.append(loss_np)  # appending loss to loss list
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)  # calculating smooth loss
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))  # metrics for loading bar
    return train_loss


# Defining one training epoch for learning not to learn model
def train_epoch_LNTL(model_encoder, model_classifier, model_aux, loader, optimizer, optimizer_aux, criterion, criterion_aux):

    # setting models to train mode
    model_encoder.train()
    model_classifier.train()
    model_aux.train()

    # empty lists for training loss and auxiliary training loss
    train_loss = []
    train_loss_aux = []

    # adding progress bar for easier monitoring during training
    bar = tqdm(loader)
    for (data, target, target_aux, target_aux2) in bar:
        if args.rulers:  # Switching to ruler labels
            target_aux = target_aux2
        # zeroing gradients
        optimizer.zero_grad()
        optimizer_aux.zero_grad()

        # sending data and targets to GPU
        data, target, target_aux = data.to(device), target.to(device), target_aux.to(device)
        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feature representation using the encoder
        logits = model_classifier(feat_out)  # using the main classifier to get output logits
        target = target.unsqueeze(1).type_as(logits)  # unsqueezing to [batch_size,1] and same dtype as logits

        # ######----------------Main Head & Pseudo Loss---------------###########

        # taking pseudo prediction from output of auxillary head (output of softmax)
        _, pseudo_pred_aux = model_aux(feat_out)

        # loss for main prediction calculated using crossentropyloss and logits output
        loss_main = criterion(logits, target)
        # pseudo auxilary loss calculated
        loss_pseudo_aux = torch.mean(torch.sum(pseudo_pred_aux * torch.log(pseudo_pred_aux), 1))
        # pseudo auxiliary loss multiplied by lambda and added to main prediction loss
        loss = loss_main + loss_pseudo_aux * args.lambdaa

        # backpropegation to calculate gradients
        loss.backward()
        # updating weights
        optimizer.step()

        # ######-------------Auxiliary Head Classifier Update------------###########

        # zeroing gradients from last step
        optimizer.zero_grad()
        optimizer_aux.zero_grad()

        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder

        # applying gradient reversal to outputted features of main network
        if args.GRL:
            feat_out = grad_reverse(feat_out)
        # getting logits from auxillary head output (gradient reversal applied ready for updating)
        logits_aux, _ = model_aux(feat_out)
        # calculating auxiliary loss
        loss_aux = criterion_aux(logits_aux, target_aux)

        # backpropegating to calculate gradients (with reversal since gradient reversal applied above)
        loss_aux.backward()
        # updating weights
        optimizer.step()
        optimizer_aux.step()

        # sending losses to cpu for printing
        loss_np = loss.detach().cpu().numpy()
        loss_aux_np = loss_aux.detach().cpu().numpy()
        # appending losses to lists
        train_loss.append(loss_np)
        train_loss_aux.append(loss_aux_np)
        # calculating smooth losses
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        smooth_loss_aux = sum(train_loss_aux[-100:]) / min(len(train_loss_aux), 100)
        bar.set_description('loss: %.5f, smth: %.5f, aux_loss: %.5f, aux_smth: %.5f' %
                            (loss_np, smooth_loss, loss_aux_np, smooth_loss_aux,))
    return train_loss, train_loss_aux


# Defining one training epoch for learning not to learn
def train_epoch_TABE(model_encoder, model_classifier, model_aux, loader, optimizer, optimizer_aux,
                     optimizer_confusion, criterion, criterion_aux):

    # setting lambda as tuning parameter for auxiliary loss
    # setting models to train mode
    model_encoder.train()
    model_classifier.train()
    model_aux.train()

    # empty lists for training loss and auxiliary training loss
    train_loss = []
    train_loss_aux = []

    # adding progress bar for easier monitoring during training
    bar = tqdm(loader)
    for (data, target, target_aux, target_aux2) in bar:
        if args.rulers:  # switching targets round if wanting to use rulers as target
            target_aux = target_aux2
        # zeroing gradients
        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_confusion.zero_grad()

        # sending data and targets to cpu
        data, target, target_aux = data.to(device), target.to(device), target_aux.to(device)
        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder
        logits = model_classifier(feat_out)  # using the main classifier to get output logits
        target = target.unsqueeze(1).type_as(logits)  # unsqueezing to[batch_size,1] and same dtype as logits

        # ######----------------Main Head & Pseudo Loss---------------###########

        loss_main = criterion(logits, target)  # using categorical cross entropy (softmax built in) to get loss

        _, output_conf = model_aux(feat_out)  # getting probabilities from auxiliary head
        # defining uniform distribution for calculating KL divergence for confusion loss
        uni_distrib = torch.FloatTensor(output_conf.size()).uniform_(0, 1)
        uni_distrib = uni_distrib.to(device)  # sending to GPU
        uni_distrib = Variable(uni_distrib)
        loss_conf = - args.alpha * (torch.sum(uni_distrib * torch.log(output_conf))) / float(output_conf.size(0))  # calculating confusion loss

        loss = loss_main + loss_conf  # adding main and confusion losses

        # backpropegation to calculate gradients
        loss.backward()
        # updating weights
        optimizer.step()
        optimizer_confusion.step()

        # ######-------------------------------Auxiliary Head Classifier Update-------------------------------###########

        # zeroing gradients from last step
        optimizer.zero_grad()
        optimizer_aux.zero_grad()

        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder

        # applying gradient reversal to outputted features of main network
        if args.GRL:
            feat_out = grad_reverse(feat_out)
        # getting logits from auxillary head output (gradient reversal applied ready for updating)
        logits_aux, _ = model_aux(feat_out)
        # calculating auxiliary loss
        loss_aux = criterion_aux(logits_aux, target_aux)

        # backpropegating to calculate gradients (with reversal since gradient reversal applied above)
        loss_aux.backward()
        # updating weights
        optimizer.step()
        optimizer_aux.step()

        # sending losses to cpu for printing
        loss_np = loss.detach().cpu().numpy()
        loss_aux_np = loss_aux.detach().cpu().numpy()
        # appending losses to lists
        train_loss.append(loss_np)
        train_loss_aux.append(loss_aux_np)
        # calculating smooth losses
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        smooth_loss_aux = sum(train_loss_aux[-100:]) / min(len(train_loss_aux), 100)
        bar.set_description('loss: %.5f, smth: %.5f, aux_loss: %.5f, aux_smth: %.5f' %
                            (loss_np, smooth_loss, loss_aux_np, smooth_loss_aux,))
    return train_loss, train_loss_aux


# Defining one training epoch for learning not to learn
def train_epoch_doubleTABE(model_encoder, model_classifier, model_aux, model_aux2, loader, optimizer, optimizer_aux,
                           optimizer_aux2, optimizer_confusion, criterion, criterion_aux, criterion_aux2):

    # setting lambda as tuning parameter for auxiliary loss
    # setting models to train mode
    model_encoder.train()
    model_classifier.train()
    model_aux.train()
    model_aux2.train()

    # empty lists for training loss and auxiliary training loss
    train_loss = []
    train_loss_aux = []
    train_loss_aux2 = []

    # adding progress bar for easier monitoring during training
    bar = tqdm(loader)
    for (data, target, target_aux, target_aux2) in bar:
        # zeroing gradients
        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_aux2.zero_grad()
        optimizer_confusion.zero_grad()

        # sending data and targets to cpu
        data, target, target_aux, target_aux2 = data.to(device), target.to(device), target_aux.to(device), target_aux2.to(device)
        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder
        logits = model_classifier(feat_out)  # using the main classifier to get output logits
        target = target.unsqueeze(1).type_as(logits)  # unsqueezing to[batch_size,1] and same dtype as logits

        # ######----------------Main Head & Confusion Loss---------------###########

        loss_main = criterion(logits, target)  # using categorical cross entropy (softmax built in) to get loss

        _, output_conf = model_aux(feat_out)  # getting probabilities from auxiliary head
        _, output_conf2 = model_aux2(feat_out)  # getting probabilities from auxiliary head
        # defining uniform distribution for calculating KL divergence for confusion loss
        uni_distrib = torch.FloatTensor(output_conf.size()).uniform_(0, 1)
        uni_distrib = uni_distrib.to(device)  # sending to GPU
        uni_distrib = Variable(uni_distrib)
        loss_conf = - args.alpha * (torch.sum(uni_distrib * torch.log(output_conf))) / float(output_conf.size(0))  # calculating confusion loss

        uni_distrib2 = torch.FloatTensor(output_conf2.size()).uniform_(0, 1)
        uni_distrib2 = uni_distrib2.to(device)  # sending to GPU
        uni_distrib2 = Variable(uni_distrib2)
        loss_conf2 = - args.alpha * (torch.sum(uni_distrib2 * torch.log(output_conf2))) / float(output_conf2.size(0))  # calculating confusion loss

        loss = loss_main + loss_conf + loss_conf2  # adding main and confusion losses

        # backpropegation to calculate gradients
        loss.backward()
        # updating weights
        optimizer.step()
        optimizer_confusion.step()

        # ######-------------------------------Auxiliary Head Classifier Update-------------------------------###########

        # zeroing gradients from last step
        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_aux2.zero_grad()

        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder

        # applying gradient reversal to outputted features of main network
        if args.GRL:
            feat_out = grad_reverse(feat_out)
        # getting logits from auxillary head output (gradient reversal applied ready for updating)
        logits_aux, _ = model_aux(feat_out)
        logits_aux2, _ = model_aux2(feat_out)
        # calculating auxiliary loss
        loss_aux = criterion_aux(logits_aux, target_aux)
        loss_aux2 = criterion_aux2(logits_aux2, target_aux2)

        aux_losses = loss_aux + loss_aux2

        # backpropegating to calculate gradients
        aux_losses.backward()

        # updating weights
        optimizer.step()
        optimizer_aux.step()
        optimizer_aux2.step()

        # sending losses to cpu for printing
        loss_np = loss.detach().cpu().numpy()
        loss_aux_np = loss_aux.detach().cpu().numpy()  # sending loss to cpu
        loss_aux2_np = loss_aux2.detach().cpu().numpy()  # sending loss to cpu

        # ------------------------------------------------------------------

        # appending losses to loss lists
        train_loss.append(loss_np)
        train_loss_aux.append(loss_aux_np)
        train_loss_aux2.append(loss_aux2_np)
        # calculating smooth losses
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        smooth_loss_aux = sum(train_loss_aux[-100:]) / min(len(train_loss_aux), 100)
        smooth_loss_aux2 = sum(train_loss_aux2[-100:]) / min(len(train_loss_aux2), 100)
        # metrics to be displayed with progress bar
        bar.set_description(
            'loss: %.5f, smth: %.5f, aux_loss: %.5f, aux_loss2: %.5f, aux_smth: %.5f, aux_smth2: %.5f' %
            (loss_np, smooth_loss, loss_aux_np, loss_aux2_np, smooth_loss_aux, smooth_loss_aux2))
    return train_loss, train_loss_aux, train_loss_aux2


# Defining one training epoch for learning not to learn
def train_epoch_BOTH(model_encoder, model_classifier, model_aux, model_aux2, loader, optimizer, optimizer_aux,
                     optimizer_aux2, optimizer_confusion, criterion, criterion_aux, criterion_aux2):
    # setting lambda as tuning parameter for auxiliary loss
    # setting models to train mode
    model_encoder.train()
    model_classifier.train()
    model_aux.train()
    model_aux2.train()

    # empty lists for training loss and auxiliary training loss
    train_loss = []
    train_loss_aux = []
    train_loss_aux2 = []

    bar = tqdm(loader)  # using tqdm to show progress bar
    for (data, target, target_aux_pre, target_aux2_pre) in bar:

        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_aux2.zero_grad()
        optimizer_confusion.zero_grad()

        if args.switch_heads:  # allowing heads to switch by switchin labels
            target_aux = target_aux2_pre
            target_aux2 = target_aux_pre
        else:
            target_aux = target_aux_pre
            target_aux2 = target_aux2_pre
        data, target, target_aux, target_aux2 = data.to(device), target.to(device), target_aux.to(
            device), target_aux2.to(device)  # sending data and targets to GPU

        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder
        logits = model_classifier(feat_out)  # using the main classifier to get output logits
        target = target.unsqueeze(1).type_as(logits)  # unsqueezing to[batch_size,1] and same dtype as logits

        # ######---------Main Head & Confusion Loss & pseudo loss---------###########

        loss_main = criterion(logits, target)  # using categorical cross entropy (softmax built in) to get loss

        _, output_conf = model_aux(feat_out)  # getting probabilities from first auxiliary head
        uni_distrib = torch.FloatTensor(output_conf.size()).uniform_(0, 1)  # calculating uniform distribution
        uni_distrib = uni_distrib.to(device)  # sending to GPU
        uni_distrib = Variable(uni_distrib)
        loss_conf = - args.alpha * (torch.sum(uni_distrib * torch.log(output_conf))) / float(
            output_conf.size(0))  # calculating confusion loss

        _, pseudo_pred_aux = model_aux(feat_out)  # taking pseudo prediction from output of auxillary head (output of softmax)
        loss_pseudo_aux = torch.mean(
            torch.sum(pseudo_pred_aux * torch.log(pseudo_pred_aux), 1))  # calculating auxiliary pseudo loss

        loss = loss_main + loss_conf + loss_pseudo_aux * args.lambdaa  # adding losses before backpropegation

        loss.backward()  # backpropegating loss to calculate gradients
        optimizer.step()  # updating weights
        optimizer_confusion.step()

        # ######----------------Auxiliary Head Classifier Update----------------###########

        # zeroing gradients from last step
        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_aux2.zero_grad()

        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder

        # applying gradient reversal to outputted features of main network
        if args.GRL:
            feat_out = grad_reverse(feat_out)
        # getting logits from auxillary head output (gradient reversal applied ready for updating)
        logits_aux, _ = model_aux(feat_out)
        logits_aux2, _ = model_aux2(feat_out)
        # calculating auxiliary loss
        if args.switch_heads:
            loss_aux = criterion_aux2(logits_aux, target_aux)  # calculating auxiliary loss
            loss_aux2 = criterion_aux(logits_aux2, target_aux2)  # calculating 2nd auxiliary loss
        else:
            loss_aux = criterion_aux(logits_aux, target_aux)  # calculating auxiliary loss
            loss_aux2 = criterion_aux2(logits_aux2, target_aux2)  # calculating 2nd auxiliary loss

        aux_losses = loss_aux + loss_aux2

        # backpropegating to calculate gradients
        aux_losses.backward()

        # updating weights
        optimizer.step()
        optimizer_aux.step()
        optimizer_aux2.step()

        # sending losses to cpu for printing
        loss_np = loss.detach().cpu().numpy()
        loss_aux_np = loss_aux.detach().cpu().numpy()  # sending loss to cpu
        loss_aux2_np = loss_aux2.detach().cpu().numpy()  # sending loss to cpu

        # ------------------------------------------------------------------

        # appending losses to loss lists
        train_loss.append(loss_np)
        train_loss_aux.append(loss_aux_np)
        train_loss_aux2.append(loss_aux2_np)
        # calculating smooth losses
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        smooth_loss_aux = sum(train_loss_aux[-100:]) / min(len(train_loss_aux), 100)
        smooth_loss_aux2 = sum(train_loss_aux2[-100:]) / min(len(train_loss_aux2), 100)
        # metrics to be displayed with progress bar
        bar.set_description(
            'loss: %.5f, smth: %.5f, aux_loss: %.5f, aux_loss2: %.5f, aux_smth: %.5f, aux_smth2: %.5f' % (
                loss_np, smooth_loss, loss_aux_np, loss_aux2_np, smooth_loss_aux, smooth_loss_aux2))
    return train_loss, train_loss_aux, train_loss_aux2


# Defining one training epoch for learning not to learn
def train_epoch_doubleLNTL(model_encoder, model_classifier, model_aux, model_aux2, loader, optimizer, optimizer_aux,
                           optimizer_aux2, criterion, criterion_aux, criterion_aux2):
    # setting lambda as tuning parameter for auxiliary loss
    # setting models to train mode
    model_encoder.train()
    model_classifier.train()
    model_aux.train()
    model_aux2.train()

    # empty lists for training loss and auxiliary training loss
    train_loss = []
    train_loss_aux = []
    train_loss_aux2 = []

    bar = tqdm(loader)  # using tqdm to show progress bar
    for (data, target, target_aux, target_aux2) in bar:

        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_aux2.zero_grad()

        data, target, target_aux, target_aux2 = data.to(device), target.to(device), target_aux.to(
            device), target_aux2.to(device)  # sending data and targets to GPU

        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder
        logits = model_classifier(feat_out)  # using the main classifier to get output logits
        target = target.unsqueeze(1).type_as(logits)  # unsqueezing to[batch_size,1] and same dtype as logits

        # ######----------------Main Head & Pseudo Losses---------------###########

        loss_main = criterion(logits, target)  # using categorical cross entropy (softmax built in) to get loss

        _, pseudo_pred_aux = model_aux(feat_out)  # taking pseudo prediction from output of auxillary head (output of softmax)
        loss_pseudo_aux = torch.mean(
            torch.sum(pseudo_pred_aux * torch.log(pseudo_pred_aux), 1))  # calculating auxiliary pseudo loss

        _, pseudo_pred_aux2 = model_aux2(feat_out)  # taking pseudo prediction from output of auxillary head (output of softmax)
        loss_pseudo_aux2 = torch.mean(
            torch.sum(pseudo_pred_aux2 * torch.log(pseudo_pred_aux2), 1))  # calculating auxiliary pseudo loss

        loss = loss_main + (loss_pseudo_aux + loss_pseudo_aux2)*args.lambdaa  # adding losses before backpropegation

        loss.backward()  # backpropegating loss to calculate gradients
        optimizer.step()  # updating weights

        # ######-------------Auxiliary Head Classifier Update------------###########

        # zeroing gradients from last step
        optimizer.zero_grad()
        optimizer_aux.zero_grad()
        optimizer_aux2.zero_grad()

        # predicting with model and getting feature maps and logits
        feat_out = model_encoder(data)  # creating feaure representation using the encoder

        # applying gradient reversal to outputted features of main network
        if args.GRL:
            feat_out = grad_reverse(feat_out)
        # getting logits from auxillary head output (gradient reversal applied ready for updating)
        logits_aux, _ = model_aux(feat_out)
        logits_aux2, _ = model_aux2(feat_out)
        # calculating auxiliary loss

        loss_aux = criterion_aux(logits_aux, target_aux)  # calculating auxiliary loss
        loss_aux2 = criterion_aux2(logits_aux2, target_aux2)  # calculating 2nd auxiliary loss

        aux_losses = loss_aux + loss_aux2

        # backpropegating to calculate gradients
        aux_losses.backward()

        # updating weights
        optimizer.step()
        optimizer_aux.step()
        optimizer_aux2.step()

        # sending losses to cpu for printing
        loss_np = loss.detach().cpu().numpy()
        loss_aux_np = loss_aux.detach().cpu().numpy()  # sending loss to cpu
        loss_aux2_np = loss_aux2.detach().cpu().numpy()  # sending loss to cpu

        # ------------------------------------------------------------------

        # appending losses to loss lists
        train_loss.append(loss_np)
        train_loss_aux.append(loss_aux_np)
        train_loss_aux2.append(loss_aux2_np)
        # calculating smooth losses
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        smooth_loss_aux = sum(train_loss_aux[-100:]) / min(len(train_loss_aux), 100)
        smooth_loss_aux2 = sum(train_loss_aux2[-100:]) / min(len(train_loss_aux2), 100)
        # metrics to be displayed with progress bar
        bar.set_description(
            'loss: %.5f, smth: %.5f, aux_loss: %.5f, aux_loss2: %.5f, aux_smth: %.5f, aux_smth2: %.5f' % (
                loss_np, smooth_loss, loss_aux_np, loss_aux2_np, smooth_loss_aux, smooth_loss_aux2))
    return train_loss, train_loss_aux, train_loss_aux2


# translations for testing (test-time augmentation)
def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model_encoder, model_classifier, loader, criterion, n_test=1, get_output=False):
    # setting models to evaluation mode
    model_encoder.eval()
    model_classifier.eval()
    # setting up storage lists
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target, _, _) in tqdm(loader):  # using tqdm for progress bar
            data, target = data.to(device), target.to(device)  # sending data to GPU
            logits = torch.zeros((data.shape[0], args.out_dim)).to(device)  # creating blank tensor for logits
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)  # creating blank tensor for probabilities
            # using translations to test on same image in different positions and using voting for consensus
            for I in range(n_test):
                feat_out = model_encoder(get_trans(data, I))  # getting feature representation from encoder
                l = model_classifier(feat_out)  # getting logits from main classifier head

                logits += l  # adding logits to logits tensor
                probs += torch.sigmoid(l)  # adding probabilities to probabilities tensor
            logits /= n_test  # dividing logits by number of tests for consensus
            probs /= n_test  # dividing probabilities by number of tests for consensus
            # appending logits, probabilities and targets to storage lists
            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())
            target = target.unsqueeze(1).type_as(logits)  # Unsqueezing to[batch_size,1] and same dtype as logits

            loss = criterion(logits, target)  # getting batch loss
            val_loss.append(loss.detach().cpu().numpy())  # getting batch validation loss

    val_loss = np.mean(val_loss)  # getting overall validation loss
    # converting to numpy
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return PROBS, TARGETS
    else:
        acc = accuracy_score(TARGETS, np.round(PROBS))  # calculating accuracy, 0.5 threshold
        auc = roc_auc_score(TARGETS, PROBS)  # calculating area under the curve
        return val_loss, acc, auc
