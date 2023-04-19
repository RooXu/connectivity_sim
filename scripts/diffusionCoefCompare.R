# Do multiple conditions 

library(tidyverse)
library(dplyr)
library(ggplot2)
library(broom)
library(ggpubr)
getwd()
rm(list = ls())

source("./scripts/rsqOverTime.R")

ipmd000 <- plotfitMatrixCondition("agent_log_FC000.*..csv",0)
ipmd020 <- plotfitMatrixCondition("agent_log_FC020.*..csv",20)
ipmd040 <- plotfitMatrixCondition("agent_log_FC040.*..csv",40)
ipmd060 <- plotfitMatrixCondition("agent_log_FC060.*..csv",60)
ipmd080 <- plotfitMatrixCondition("agent_log_FC080.*..csv",80)
ipmd100 <- plotfitMatrixCondition("agent_log_FC100.*..csv",100)

fits <- bind_rows(ipmd000[2],ipmd020[2],ipmd040[2],ipmd060[2],ipmd080[2],ipmd100[2])

ggplot(fits,aes(x=density,group=density,y=slope,fill=density))+
  geom_boxplot()

my_comparisons = list(c("0","100"),c("0","80"),c("0","60"),c("0","40"),c("0","20"),
                   c("20","100"),c("20","80"),c("20","60"),c("20","40"),
                   c("40","100"),c("40","80"),c("40","60"), 
                   c("60","100"),c("60","80"),
                   c("80","100"))
ggboxplot(fits,x="density",y="slope",color = "density", palette = "jco",add = 'jitter') + 
  stat_compare_means(comparisons = my_comparisons)
