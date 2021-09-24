import os
import matplotlib.pyplot as plt
import time
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from globalbaz import args, DP, device
from dataset import *
from models import *
from test import *
from train_epoch_variations import *


# Setting seeds for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Function used to plot the curves for loss and accuracy:
def plot_curves(auc):
    # Plotting the AUC curve:
    plt.style.use('ggplot')
    plt.title('Validation AUC')
    plt.ylim(0, 1)
    plt.xlim(0, args.n_epochs)
    plt.xticks(range(0, args.n_epochs))
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    # Plotting the test accuracy (red):
    plt.plot(auc, color='red', label='test')
    return plt
    print('Done!')


# Main training function
def run(df_train, transforms_train, transforms_val, transforms_marked,
        criterion, criterion_aux, criterion_aux2, fold=None):
    if args.cv:  # If using k-fold cross-validation
        # specifying fold to leave out
        i_fold = fold
        # subsetting data to leave out validation fold
        # DEBUG mode reduces epoch to 3 batches
        if args.DEBUG:
            df_this = df_train[df_train['fold'] != i_fold].sample(args.batch_size * 3)
            df_valid = df_train[df_train['fold'] == i_fold].sample(args.batch_size * 3)
        else:
            df_this = df_train[df_train['fold'] != i_fold]
            df_valid = df_train[df_train['fold'] == i_fold]
    else:
        if args.DEBUG:
            df_this = df_train.sample(args.batch_size * 3)
        else:
            df_this = df_train

    # Setting different number of units for fully connected layer based on feature extractor output
    if args.arch == 'resnet101' or args.arch == 'resnext101' or args.arch == 'inception':
        in_ch = 2048
    elif args.arch == 'densenet':
        in_ch = 2208
    else:
        in_ch = 1536

    # Loading training data
    dataset_train = SIIMISICDataset(df_this, 'train', 'train', transform=transforms_train, transform2=transforms_marked)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               sampler=RandomSampler(dataset_train),
                                               num_workers=args.num_workers, drop_last=True)
    if args.cv:
        # Loading validation data
        dataset_valid = SIIMISICDataset(df_valid, 'train', 'val', transform=transforms_val,
                                        transform2=transforms_marked)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, drop_last=True)

    # defining feature extractor model and sending to gpu
    if args.arch == 'enet':
        model_encoder = enetv2(args.enet_type)
    if args.arch == 'resnet101':
        model_encoder = ResNet101()
    if args.arch == 'resnext101':
        model_encoder = ResNext101()
    if args.arch == 'densenet':
        model_encoder = DenseNet()
    if args.arch == 'inception':
        model_encoder = Inception()

    model_classifier = ClassificationHead(out_dim=args.out_dim, in_ch=in_ch)  # Creating main classification head
    if DP:  # Parallelising if number of GPUs allows
        model_encoder = nn.DataParallel(model_encoder)
        model_classifier = nn.DataParallel(model_classifier)
    model_encoder = model_encoder.to(device)  # Sending feature extractor to GPU
    model_classifier = model_classifier.to(device)  # Sending classifier head to GPU

# --------Setting auxiliary heads if using debiasing------------------------
    # Defining 1st auxiliary head to be used across all debiasing configs
    if args.debias_config != 'baseline':
        if args.deep_aux:
            model_aux = AuxiliaryHead2(num_aux=args.num_aux, in_ch=in_ch)
        else:
            model_aux = AuxiliaryHead(num_aux=args.num_aux, in_ch=in_ch)  # defining auxiliary head
        if DP:
            model_aux = nn.DataParallel(model_aux)  # for running on multiple GPUs
        model_aux = model_aux.to(device)  # sending auxiliary head to GPU
        # Defining second auxiliary heads if using double header
        if args.debias_config == 'doubleTABE' or args.debias_config == 'both' or args.debias_config == 'doubleLNTL':
            if args.deep_aux:
                model_aux2 = AuxiliaryHead2(num_aux=args.num_aux2, in_ch=in_ch)  # defining 2nd auxiliary head
            else:
                model_aux2 = AuxiliaryHead(num_aux=args.num_aux2, in_ch=in_ch)  # defining 2nd auxiliary head
            if DP:
                model_aux2 = nn.DataParallel(model_aux2)  # for running on multiple GPUs
            model_aux2 = model_aux2.to(device)  # sending to GPU

    # Defining main optimizer used accross all models
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, (list(model_encoder.parameters()) + list(model_classifier.parameters()))),
        lr=args.lr_base, momentum=args.momentum)

# --------Setting auxiliary optimisers based on debias config----------------
    # Defining auxiliary optimizer for LNTL
    if args.debias_config == 'LNTL':
        # defining auxiliary optimiser (LNTL)
        optimizer_aux = optim.SGD(model_aux.parameters(), lr=args.lr_base, momentum=args.momentum)

    # defining auxiliary optimisers for TABE
    if args.debias_config == 'TABE':
        optimizer_confusion = optim.SGD(model_encoder.parameters(), lr=args.lr_class,
                                        momentum=args.momentum)  # Defining confusion optimiser (boosted encoder optimiser)
        optimizer_aux = optim.SGD(model_aux.parameters(), lr=args.lr_class, momentum=args.momentum)  # defining auxiliary classification optimiser

    # defining case where two auxiliary heads present, both TABE
    if args.debias_config == 'doubleTABE':
        optimizer_confusion = optim.SGD(model_encoder.parameters(), lr=args.lr_class,
                                        momentum=args.momentum)  # defining confusion optimiser (boosted encoder optimiser)
        optimizer_aux = optim.SGD(model_aux.parameters(), lr=args.lr_class,
                                  momentum=args.momentum)  # defining classification optimiser for 1st aux head
        optimizer_aux2 = optim.SGD(model_aux2.parameters(), lr=args.lr_class,
                                   momentum=args.momentum)  # defining classification optimiser for 2nd aux head

    # Defining case where first aux head is LNTL and second is TABE
    if args.debias_config == 'both':
        # Defining 1st auxiliary optimiser (LNTL)
        optimizer_aux = optim.SGD(model_aux.parameters(), lr=args.lr_base, momentum=args.momentum)
        optimizer_confusion = optim.SGD(model_encoder.parameters(), lr=args.lr_class,
                                        momentum=args.momentum)  # defining confusion optimiser (boosted encoder optimiser)
        optimizer_aux2 = optim.SGD(model_aux2.parameters(), lr=args.lr_class,
                                   momentum=args.momentum)  # defining optimiser for 2nd auxiliary head

    # Defining case where first aux head is LNTL and second is TABE
    if args.debias_config == 'doubleLNTL':
        # defining 1st auxiliary optimiser (LNTL)
        optimizer_aux = optim.SGD(model_aux.parameters(), lr=args.lr_base, momentum=args.momentum)
        optimizer_aux2 = optim.SGD(model_aux2.parameters(), lr=args.lr_base, momentum=args.momentum)  # defining optimiser for 2nd auxiliary head

# ---------------------------------------------------------------------------------------

    # defining variables to save scores to
    auc_max = 0.
    val_losses = []
    auc_lst = [0]

    if args.cv:
        # setting up model filenames for saving when auc improves
        encoder_file_cv = f'{args.model_dir}/{args.test_no}/encoder_best_fold{i_fold}.pth'
        classifier_file_cv = f'{args.model_dir}/{args.test_no}/classifier_best_fold{i_fold}.pth'

    # looping through epochs to train
    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)  # printing time and epoch number
        # Running different epoch variations based on debias config argument
        if args.debias_config == 'baseline':
            train_loss = train_epoch_baseline(model_encoder, model_classifier, train_loader, optimizer, criterion)
        if args.debias_config == 'LNTL':
            train_loss, train_loss_aux = train_epoch_LNTL(model_encoder, model_classifier, model_aux, train_loader,
                                                          optimizer, optimizer_aux, criterion, criterion_aux)
        if args.debias_config == 'TABE':
            train_loss, train_loss_aux = train_epoch_TABE(model_encoder, model_classifier, model_aux, train_loader,
                                                          optimizer, optimizer_aux, optimizer_confusion, criterion,
                                                          criterion_aux)
        if args.debias_config == 'both':
            train_loss, train_loss_aux, train_loss_aux2 = train_epoch_BOTH(model_encoder, model_classifier, model_aux,
                                                                           model_aux2, train_loader, optimizer,
                                                                           optimizer_aux, optimizer_aux2,
                                                                           optimizer_confusion, criterion,
                                                                           criterion_aux, criterion_aux2)
        if args.debias_config == 'doubleTABE':
            train_loss, train_loss_aux, train_loss_aux2 = train_epoch_doubleTABE(
                model_encoder, model_classifier, model_aux, model_aux2, train_loader, optimizer, optimizer_aux,
                optimizer_aux2, optimizer_confusion, criterion, criterion_aux, criterion_aux2)
        if args.debias_config == 'doubleLNTL':
            train_loss, train_loss_aux, train_loss_aux2 = train_epoch_doubleLNTL(model_encoder, model_classifier,
                                                                                 model_aux, model_aux2, train_loader,
                                                                                 optimizer, optimizer_aux,
                                                                                 optimizer_aux2, criterion,
                                                                                 criterion_aux, criterion_aux2)
        if args.cv:
            # validation epoch for val scores
            val_loss, acc, auc = val_epoch(model_encoder, model_classifier, valid_loader, criterion)
            val_losses.append(val_loss)
            auc_lst.append(auc)
            # getting metrics depending on model type
            if args.debias_config == 'baseline':
                content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch},' \
                                               f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                               f' train loss: {np.mean(train_loss):.5f},' \
                                               f' valid loss: {(val_loss):.5f},' \
                                               f' acc: {(acc):.4f},' \
                                               f' auc: {(auc):.6f}.'

            if args.debias_config == 'LNTL' or args.debias_config == 'TABE':
                content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch},' \
                                               f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                               f' train loss: {np.mean(train_loss):.5f},' \
                                               f' train loss aux: {np.mean(train_loss_aux):.5f},' \
                                               f' valid loss: {(val_loss):.5f},' \
                                               f' acc: {(acc):.4f},' \
                                               f' auc: {(auc):.6f}.'

            if args.debias_config == 'both' or args.debias_config == 'doubleTABE' or args.debias_config == 'doubleLNTL':
                content = time.ctime() + ' ' + f'Epoch {epoch},' \
                                               f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                               f' train loss: {np.mean(train_loss):.5f},' \
                                               f' train loss aux: {np.mean(train_loss_aux):.5f},' \
                                               f' train loss aux2: {np.mean(train_loss_aux2):.5f},' \
                                               f' valid loss: {(val_loss):.5f},' \
                                               f' acc: {(acc):.4f},' \
                                               f' auc: {(auc):.6f}.'

            if args.cv:  # If doing k-fold cross validation
                # saving model if score exceeds best model
                if auc > auc_max:
                    print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
                    torch.save(model_encoder.state_dict(), encoder_file_cv)
                    torch.save(model_classifier.state_dict(), classifier_file_cv)
                    auc_max = auc
        else:
            if args.debias_config == 'baseline':
                content = time.ctime() + ' ' + f'Epoch {epoch},' \
                                               f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                               f' train loss: {np.mean(train_loss):.5f}'

            if args.debias_config == 'LNTL' or args.debias_config == 'TABE':
                content = time.ctime() + ' ' + f'Epoch {epoch},' \
                                               f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                               f' train loss: {np.mean(train_loss):.5f},' \
                                               f' train loss aux: {np.mean(train_loss_aux):.5f}'

            if args.debias_config == 'both' or args.debias_config == 'doubleTABE' or args.debias_config == 'doubleLNTL':
                content = time.ctime() + ' ' + f'Epoch {epoch},' \
                                               f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                               f' train loss: {np.mean(train_loss):.5f},' \
                                               f' train loss aux: {np.mean(train_loss_aux):.5f},' \
                                               f' train loss aux2: {np.mean(train_loss_aux2):.5f}'

        print(content)

        # writing metrics to text file
        with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}.txt'), 'a') as appender:
            appender.write(content + '\n')

        # saving model if training on full data
        if not args.cv:
            if epoch % args.save_epoch == 0:
                torch.save(
                    {'model_state_dict': model_encoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                     'epoch': epoch}, os.path.join(args.model_dir,
                                                   f'{args.test_no}/encoder_all_data_Test{args.test_no}.pth'))
                torch.save(model_classifier.state_dict(),
                           os.path.join(args.model_dir, f'{args.test_no}/classifier_all_data_Test{args.test_no}.pth'))
                print('model saved')

    # saving model if cross validating
    if args.cv:
        # plotting learning curves and saving for examination to decide optimal epoch
        val_curve = plot_curves(auc_lst)
        val_curve.savefig(f'{args.plot_dir}/{args.test_no}/val_curve_Test{args.test_no}_fold{fold}.pdf')
        # saving main model at end of fold
        torch.save(model_encoder.state_dict(),
                   os.path.join(args.model_dir, f'{args.test_no}/encoder_Test{args.test_no}_fold{i_fold}.pth'))
        torch.save(model_classifier.state_dict(),
                   os.path.join(args.model_dir, f'{args.test_no}/classifier_Test{args.test_no}_fold{i_fold}.pth'))

    # plotting learning curves and saving for examination to decide optimal epoch
    val_curve = plot_curves(auc_lst)
    val_curve.savefig(f'{args.plot_dir}/{args.test_no}/val_curve_Test{args.test_no}.pdf')


def main():

    # writing arguments to text file
    with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}.txt'),
              'a') as appender:
        appender.write(str(args) + '\n')

    # Loading training, validation and test dataframes
    df_train, df_val, df_test_blank, df_test_marked, df_test_rulers, df_test_atlasD, df_test_atlasC, df_test_ASAN,\
        df_test_MClassD, df_test_MClassC, df_34, df_56, mel_idx = get_df()

    # Selecting test data based on experiment
    if args.sktone and args.split_skin_types:
        df_test_lst = [df_34, df_56]
    elif args.tune:
        df_test_lst = [df_val]
    elif args.heid_test_marked:
        df_test_lst = [df_test_blank, df_test_marked]
    elif args.heid_test_rulers:
        df_test_lst = [df_test_blank, df_test_rulers]
    else:
        df_test_lst = [df_test_atlasD, df_test_atlasC, df_test_ASAN, df_test_MClassD, df_test_MClassC]

    criterion, criterion_aux, criterion_aux2 = criterion_func(df_train)

    transforms_marked, transforms_train, transforms_val = get_transforms()

    # if doing k-fold cross-validation
    if args.cv:
        scores = []
        for fold in range(5):
            run(df_train, transforms_train, transforms_val, transforms_marked, criterion, criterion_aux,
                criterion_aux2, fold=fold)
        print(scores)
        cv_acc, cv_auc, cv_auc_rpf = cv_scores(df_train, mel_idx, transforms_val, transforms_marked)
        cv_scores_content = f'cv_acc: {cv_acc}, cv_auc: {cv_auc}, cv_auc_rpf: {cv_auc_rpf}'
        with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}.txt'),
                  'a') as appender:
            appender.write(cv_scores_content + '\n')

    else:
        if not args.test_only:  # Skipping training if test_only
            run(df_train, transforms_train, transforms_val,
                transforms_marked, criterion, criterion_aux, criterion_aux2)
        roc_plt_lst = []  # list of tuples of metrics needed to plot ROC curves
        for index, df in enumerate(df_test_lst):
            fpr, tpr, a_u_c, sensitivity, specificity = test(index, df, mel_idx, transforms_val)
            roc_plt_lst.append((fpr, tpr, a_u_c, sensitivity, specificity))
            saliency(index, df, transforms_val)  # Plotting saliency maps
        ROC_curve(roc_plt_lst)  # Plotting ROC curves
        # Pickling info needed to plot custom ROC plots with misc_code/ROC_plots.py
        with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}_roc_plt_lst.pkl'),
                  'wb') as f:
            pickle.dump(roc_plt_lst, f)


if __name__ == '__main__':

    # Making directories to save results and weights to
    os.makedirs(f'{args.model_dir}/{args.test_no}', exist_ok=True)
    os.makedirs(f'{args.log_dir}/{args.test_no}', exist_ok=True)
    os.makedirs(f'{args.plot_dir}/{args.test_no}', exist_ok=True)

    # Printing out configuration at start of training to make sure all correct
    print(args)
    print('------------------------------------')
    print(f'Model architechture: {args.arch}')
    print(f'Debiasing configuration: {args.debias_config}')
    print(f'Training dataset: {args.dataset}')
    print('------------------------------------')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    set_seed(seed=args.seed)  # Setting seeds for reproducibility

    main()
