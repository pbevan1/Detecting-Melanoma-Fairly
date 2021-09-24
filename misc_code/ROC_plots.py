import pickle
import matplotlib.pyplot as plt
import os

# write a list (test_nos) of the test numbers you want to plot against each other on different datasets. Write another
# list with the corresponding names and pass these into the below function. This will generate plots that compare
# configurations on the same dataset


def ROC_curve_custom(conf_lst, test_nos, experiment, set='gen'):
    # Plot ROC curve
    roc_plt_lst = []
    if set == 'gen':
        dset_lst = ['Atlas_Dermoscopic', 'Atlas_Clinical', 'ASAN', 'MClass_Dermoscopic', 'MClass_Clinical']
    else:
        dset_lst = ['Heid_Plain', 'Heid_Marked', 'Heid_Rulers']

    colours = ["#D55E00", "#0072B2", "#E69F00", "#CC79A7"]

    fig = plt.figure()
    plt.style.use('ggplot')
    plt.use_sticky_edges = False
    plt.margins(0.005)
    for i, num in enumerate(test_nos):
        with open(f'../results/logs/{num}/log_Test{num}_roc_plt_lst.pkl', 'rb') as f:
            roc_plt_lst.append(pickle.load(f))
    for set_i, set_name in enumerate(dset_lst):
        for i, roc in enumerate(roc_plt_lst):
            for index, thruple in enumerate(roc[set_i:set_i+1]):
                # for conf_i,_ in enumerate(conf_lst):
                fpr, tpr, a_u_c, _, _ = thruple
                plt.plot(fpr, tpr, label=f'{conf_lst[i]} (AUC = {round(a_u_c,3)})', color=colours[i])
        if set_i == 3:
            plt.plot(0.3998, 0.7411, 'k+', label=f'Dermatologist (AUC* = {0.671})', markersize=10, mew=2)
        if set_i == 4:
            plt.plot(0.3563, 0.894, 'k+', label=f'Dermatologist (AUC* = {0.769})', markersize=10, mew=2)

        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (1-specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (sensitivity)', fontsize=14)
        plt.title(f'ROC: {set_name}')
        plt.legend(loc="lower right")
        plt.rc('legend', fontsize=12)  # legend fontsize
        os.makedirs(f'../results/plots/comparison_AUC/{experiment}', exist_ok=True)
        fig.savefig(f'../results/plots/comparison_AUC/{experiment}/ROC_curve_{set_name}.pdf')
        # plt.show()
        plt.clf()


if __name__ == '__main__':

    conf_lst = ['Baseline', 'LNTL', 'TABE', 'CLGR']

    # # Getting plots for marking removal
    # test_nos = [249, 228, 229, 230]
    # ROC_curve_custom(conf_lst, test_nos, 'marking_removal', 'marking')
    #
    # # Getting plots for ruler removal
    # test_nos = [252, 232, 233, 234]
    # ROC_curve_custom(conf_lst, test_nos, 'ruler_removal', 'ruler')
    #
    # #getting plots for domain generalisation
    # test_nos = [281, 282, 283, 284]
    # ROC_curve_custom(conf_lst, test_nos, 'domain_generalisation')

    #getting plots for skin tone removal
    test_nos = [348, 349, 350, 351]
    ROC_curve_custom(conf_lst, test_nos, 'isic_sktone')
    #
    # #getting plots for domain generalisation
    # test_nos = [281, 283]
    # ROC_curve_custom(conf_lst, test_nos, 'domain_generalisation_tabe')