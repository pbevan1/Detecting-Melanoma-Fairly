library(tidyverse)
library(gridExtra)

#defining colourblind palette for use in plots
cbbPalette = c("#009E73", "#56B4E9", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", '#000000', "#E69F00")

#plotting class distribution of isic 2020 and 2017 data
df = read.csv(file = '../data/csv/plotting_csv/df_17_20_plotting.csv')
class_dist = ggplot(df,aes(x=factor(benign_malignant), fill=benign_malignant, label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#E69F00", "#0072B2")) + 
  labs(fill='Benign/\nMalignant') + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/class_dist.pdf", plot = class_dist)

#plotting class distribution of Atlas data
df = read.csv(file = '../data/csv/atlas_processed.csv')
class_dist = ggplot(df,aes(x=target, fill=target, label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#E69F00", "#0072B2")) + 
  labs(fill='Benign/\nMalignant') + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/atlas_class_dist.pdf", plot = class_dist)

#plotting class distribution of asan data
df = read.csv(file = '../data/csv/asan.csv')
class_dist = ggplot(df,aes(x=target, fill=target, label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#E69F00", "#0072B2")) + 
  labs(fill='Benign/\nMalignant') + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/asan_class_dist.pdf", plot = class_dist)

#plotting class distribution of MClassD data
df = read.csv(file = '../data/csv/MClassD.csv')
class_dist = ggplot(df,aes(x=target, fill=target, label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#E69F00", "#0072B2")) + 
  labs(fill='Benign/\nMalignant') + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/MClassD_class_dist.pdf", plot = class_dist)

#plotting class distribution of MClassC data
df = read.csv(file = '../data/csv/MClassC.csv')
class_dist = ggplot(df,aes(x=target, fill=target, label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#E69F00", "#0072B2")) + 
  labs(fill='Benign/\nMalignant') + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/MClassC_class_dist.pdf", plot = class_dist)

#plotting distribution of ruler and markings in ISIC
df = read.csv(file = '../data/csv/plotting_csv/df_17_20_plotting.csv')
ruler_dist = ggplot(df,aes(x=factor(scale), fill=factor(scale), label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(labels = c('no', 'yes'), values = c("#0072B2", "#D55E00"))+
  labs(x = 'Ruler Present', y = 'Count', fill = '') + 
  scale_x_discrete(labels= c('no', 'yes')) + 
  labs(fill='Ruler\nPresent') + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/ruler_dist.pdf", plot = ruler_dist)

marked_dist = ggplot(df,aes(x=factor(marked), fill=factor(marked), label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(labels = c('no', 'yes'), values = c("#0072B2", "#D55E00"))+
  labs(x = 'Surgical Marking Present', y = 'Count', fill = '') + 
  scale_x_discrete(labels= c('no', 'yes')) + 
  labs(fill='Markings\nPresent') + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/marked_dist.pdf", plot = marked_dist)


#plotting distribution of instruments
df_instr_plt = read.csv(file = '../data/csv/plotting_csv/df_main_instr_17_20.csv')
instruments = ggplot(df_instr_plt,aes(x=reorder(size,size, function(x)-length(x)), fill=size, label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#009E73", "#56B4E9", "#F0E442", "#0072B2",
                               "#D55E00", "#CC79A7", '#000000', "#E69F00")) + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/figs/instruments.pdf", plot = instruments)


#plotting distribution of individual conditions
df_diag = read.csv(file = '../data/csv/plotting_csv/df_17_20_diagnosis.csv')
conditions = ggplot(df_diag,aes(x=reorder(diagnosis,diagnosis, function(x)-length(x)), fill=diagnosis, label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2) + 
  scale_fill_manual(values = c("#009E73", "#56B4E9", "#F0E442", "#0072B2",
                               "#D55E00", "#CC79A7"))
ggsave(filename="../results/plots/figs/conditions.pdf", plot = conditions)


#plotting mark removal with 6 random seeds
df_bar_marked = read.csv(file = '../data/csv/output_csv/bar_marked.csv')
df_bar_marked$model = factor(df_bar_marked$model, levels = c('baseline', 'LNTL', 'TABE', 'CLGR'))
df_bar_marked$test = factor(df_bar_marked$test, levels = c('plain', 'marked'))
bar_marked = ggplot(df_bar_marked, aes(x=test, y=AUC_mean, fill=model)) +
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
  geom_errorbar(aes(ymin=AUC_mean-AUC_std, ymax=AUC_mean+AUC_std), width=.2,position=position_dodge(.9))+
  scale_fill_manual(values = c("#D55E00", "#0072B2", "#E69F00", "#CC79A7")) +
  labs(title="", x="test set", y = "AUC") +
  coord_cartesian(ylim=c(0.80,1.0)) + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/Figs/bar_marked2.pdf", plot = bar_marked,
       width = 4.97, 
       height = 3.64, 
       units = "in")

#plotting ruler removal with 6 random seeds
df_bar_rulers = read.csv(file = '../data/csv/output_csv/bar_rulers.csv')
df_bar_rulers$model = factor(df_bar_rulers$model, levels = c('baseline', 'LNTL', 'TABE', 'CLGR'))
df_bar_rulers$test = factor(df_bar_rulers$test, levels = c('plain', 'rulers'))
bar_rulers = ggplot(df_bar_rulers, aes(x=test, y=AUC_mean, fill=model)) +
  geom_bar(position=position_dodge(), stat="identity", colour='black') +
  geom_errorbar(aes(ymin=AUC_mean-AUC_std, ymax=AUC_mean+AUC_std), width=.2,position=position_dodge(.9))+
  scale_fill_manual(values = c("#D55E00", "#0072B2", "#E69F00", "#CC79A7")) +
  labs(title="", x="test set", y = "AUC") +
  coord_cartesian(ylim=c(0.80,1.0)) + 
  theme(text = element_text(size=14))
ggsave(filename="../results/plots/Figs/bar_rulers2.pdf", plot = bar_rulers,
       width = 4.97, 
       height = 3.64, 
       units = "in")

#plotting Fitzpatrick skin types distribution in Fitzpatrick17k data based on automated labels
df_fitz = read.csv(file = '../data/csv/fitzpatrick17k.csv')
df_fitz = df_fitz[df_fitz$fitzpatrick>0, ]
fitz = ggplot(df_fitz,aes(x=fitzpatrick, fill=factor(fitzpatrick), label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#009E73", "#56B4E9", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", '#000000', "#E69F00")) + 
  scale_x_continuous(breaks=1:6, labels=c(1, 2, 3, 4, 5, 6)) + 
  labs(fill='Fitzpatrick\nSkin Type') + 
  theme(text = element_text(size=14)) + 
  labs(x = 'Fitzpatrick Skin Type', y = 'Count', fill = '')
ggsave(filename="../results/plots/figs/fitz.pdf", plot = fitz)

#plotting Fitzpatrick skin types distribution in ISIC data based on automated labels
df = read.csv(file = '../data/csv/plotting_csv/df_17_20_plotting.csv')
isic_fitz = ggplot(df,aes(x=fitzpatrick, fill=factor(fitzpatrick), label = scales::percent(prop.table(stat(count)))))+
  geom_bar(position="dodge", colour='black')+ 
  labs(title="", x="", y = "Count")+
  geom_text(stat='count',position=position_dodge(0.9),vjust=-0.2)+
  scale_fill_manual(values = c("#009E73", "#56B4E9", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", '#000000', "#E69F00")) + 
  scale_x_continuous(breaks=1:6, labels=c(1, 2, 3, 4, 5, 6)) + 
  labs(fill='Fitzpatrick\nSkin Type') + 
  theme(text = element_text(size=14)) + 
  labs(x = 'Fitzpatrick Skin Type', y = 'Count', fill = '')
ggsave(filename="../results/plots/figs/isic_fitz.pdf", plot = isic_fitz)
