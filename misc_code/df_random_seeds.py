import pickle
import numpy as np
import pandas as pd


# Separating out AUC and specificity from pickle object
def auc_lst(test_nos, dset_lst):
    metric_lst = []
    auc_lst = []
    spec_lst = []

    for i, num in enumerate(test_nos):
        with open(f'../results/logs/{num}/log_Test{num}_roc_plt_lst.pkl', 'rb') as f:
            metric_lst.append(pickle.load(f))

    for set_i, set_name in enumerate(dset_lst):
        for i, roc in enumerate(metric_lst):
            for index, thruple in enumerate(roc[set_i:set_i + 1]):
                # for conf_i,_ in enumerate(conf_lst):
                _, _, a_u_c, _, specificity = thruple
                auc_lst.append(a_u_c)
                spec_lst.append(specificity)
    return auc_lst, spec_lst


if __name__ == '__main__':
    # Surgical markings
    # Getting AUC values for each random seed of each model type (replace test numbers as necessary)
    AUC_baseline, spec_baseline = auc_lst([249, 253, 257, 261, 265, 269], ['Heid_Blank', 'Heid_Marked'])
    AUC_LNTL, spec_LNTL = auc_lst([228, 254, 258, 262, 266, 270], ['Heid_Blank', 'Heid_Marked'])
    AUC_TABE, spec_TABE = auc_lst([229, 255, 259, 263, 267, 271], ['Heid_Blank', 'Heid_Marked'])
    AUC_CLGR, spec_CLGR = auc_lst([230, 256, 260, 264, 268, 272], ['Heid_Blank', 'Heid_Marked'])

    # Getting mean and standard deviation of AUC and specificity for baseline, LNTL, TABE and CLGR models
    # Tested on unbiased images
    df_plain = pd.DataFrame([[np.mean(AUC_baseline[0:6]), np.std(AUC_baseline[0:6]), np.mean(spec_baseline[0:6]), np.std(spec_baseline[0:6])],
                             [np.mean(AUC_LNTL[0:6]), np.std(AUC_LNTL[0:6]), np.mean(spec_LNTL[0:6]), np.std(spec_LNTL[0:6])],
                             [np.mean(AUC_TABE[0:6]), np.std(AUC_TABE[0:6]), np.mean(spec_TABE[0:6]), np.std(spec_TABE[0:6])],
                             [np.mean(AUC_CLGR[0:6]), np.std(AUC_CLGR[0:6]), np.mean(spec_CLGR[0:6]), np.std(spec_CLGR[0:6])]],
                            columns=['AUC_mean', 'AUC_std', 'specificity_mean', 'specificity_std'])
    df_plain['test'] = 'plain'
    # tested on biased images (marked)
    df_marked = pd.DataFrame([[np.mean(AUC_baseline[6:]), np.std(AUC_baseline[6:]), np.mean(spec_baseline[6:]), np.std(spec_baseline[6:])],
                             [np.mean(AUC_LNTL[6:]), np.std(AUC_LNTL[6:]), np.mean(spec_LNTL[6:]), np.std(spec_LNTL[6:])],
                             [np.mean(AUC_TABE[6:]), np.std(AUC_TABE[6:]), np.mean(spec_TABE[6:]), np.std(spec_TABE[6:])],
                             [np.mean(AUC_CLGR[6:]), np.std(AUC_CLGR[6:]), np.mean(spec_CLGR[6:]), np.std(spec_CLGR[6:])]],
                             columns=['AUC_mean', 'AUC_std', 'specificity_mean', 'specificity_std'])
    df_marked['test'] = 'marked'
    # df = df_blank.join(df_marked)
    df = pd.concat([df_plain, df_marked])
    df['model'] = ['baseline', 'LNTL', 'TABE', 'CLGR', 'baseline', 'LNTL', 'TABE', 'CLGR']

    # Saving to csv for plotting with ggplot in r
    df.to_csv('../data/csv/output_csv/bar_marked.csv', index=False)

    # Rulers
    # Getting AUC values for each random seed of each model type (replace test numbers as necessary)
    AUC_baseline, spec_baseline = auc_lst([252, 295, 299, 303, 307, 311], ['Heid_Blank', 'Heid_Marked', 'Heid_Rulers'])
    AUC_LNTL, spec_LNTL = auc_lst([232, 296, 300, 305, 308, 312], ['Heid_Blank', 'Heid_Marked', 'Heid_Rulers'])
    AUC_TABE, spec_TABE = auc_lst([233, 297, 301, 305, 309, 313], ['Heid_Blank', 'Heid_Marked', 'Heid_Rulers'])
    AUC_CLGR, spec_CLGR = auc_lst([234, 298, 302, 306, 310, 314], ['Heid_Blank', 'Heid_Marked', 'Heid_Rulers'])

    # Getting mean and standard deviation of AUC and specificity for baseline, LNTL, TABE and CLGR models
    # Tested on unbiased images
    df_plain1 = pd.DataFrame([[np.mean(AUC_baseline[0:6]), np.std(AUC_baseline[0:6]), np.mean(spec_baseline[0:6]), np.std(spec_baseline[0:6])],
                             [np.mean(AUC_LNTL[0:6]), np.std(AUC_LNTL[0:6]), np.mean(spec_LNTL[0:6]), np.std(spec_LNTL[0:6])],
                             [np.mean(AUC_TABE[0:6]), np.std(AUC_TABE[0:6]), np.mean(spec_TABE[0:6]), np.std(spec_TABE[0:6])],
                             [np.mean(AUC_CLGR[0:6]), np.std(AUC_CLGR[0:6]), np.mean(spec_CLGR[0:6]), np.std(spec_CLGR[0:6])]],
                             columns=['AUC_mean', 'AUC_std', 'specificity_mean', 'specificity_std'])
    df_plain1['test'] = 'plain'
    # tested on biased images (rulers)
    df_rulers = pd.DataFrame([[np.mean(AUC_baseline[12:]), np.std(AUC_baseline[12:]), np.mean(spec_baseline[12:]), np.std(spec_baseline[12:])],
                             [np.mean(AUC_LNTL[12:]), np.std(AUC_LNTL[12:]), np.mean(spec_LNTL[12:]), np.std(spec_LNTL[12:])],
                             [np.mean(AUC_TABE[12:]), np.std(AUC_TABE[12:]), np.mean(spec_TABE[12:]), np.std(spec_TABE[12:])],
                             [np.mean(AUC_CLGR[12:]), np.std(AUC_CLGR[12:]), np.mean(spec_CLGR[12:]), np.std(spec_CLGR[12:])]],
                             columns=['AUC_mean', 'AUC_std', 'specificity_mean', 'specificity_std'])
    df_rulers['test'] = 'rulers'
    df1 = pd.concat([df_plain1, df_rulers])
    df1['model'] = ['baseline', 'LNTL', 'TABE', 'CLGR', 'baseline', 'LNTL', 'TABE', 'CLGR']

    # Saving results to csv for plotting with ggplot in r
    df1.to_csv('../data/csv/output_csv/bar_rulers.csv', index=False)

    print(df1)
