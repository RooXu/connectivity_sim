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
fits$density<-fits$density %>% as.factor()
ggplot(fits,aes(x=density,group=density,y=slope,col=density))+
  geom_boxplot()+
  geom_point()+
  theme_classic()

my_comparisons = list(c("0","100"),c("0","80"),c("0","60"),c("0","40"),c("0","20"),
                   c("20","100"),c("20","80"),c("20","60"),c("20","40"),
                   c("40","100"),c("40","80"),c("40","60"), 
                   c("60","100"),c("60","80"),
                   c("80","100"))
ggboxplot(fits,x="density",y="slope",color = "density", palette = "jco",add = 'jitter') + 
  stat_compare_means(comparisons = my_comparisons)

ggboxplot(fits,x="density",y="slope",color = "density", palette = "jco",add = 'jitter') +      # Add global p-value
  stat_compare_means(label = "p.signif", method = "t.test",
                     ref.group = "0")
my_theme <- theme_pubr(
  base_size = 12,
  base_family = "sans",
  border = FALSE,
  margin = TRUE,
  legend = c("top", "bottom", "left", "right", "none"),
  x.text.angle = 0
)

slopePlot<-ggline(fits,
       x="density",
       y="slope",
       add = c("mean_sd",'point'),
       add.params = list(color ='black',size = 0.5),  
       point.color='blue',
       shape = 22,
       stroke=1,
       palette = "jco",
       plot_type = 'b',
       ylab = 'alpha',
       xlab = "Matrix Occupation Density",
       ylim = c(0.5,1.5)) +
  stat_compare_means(label = "p.signif", 
                     method = "t.test",
                     ref.group = "0")+ylab(expression(alpha))+
  my_theme   
 
compare_means(slope ~ density, ref.group = "0", data = fits,method = 't.test')  
 
interceptPlot<-ggline(fits,x="density",y="intercept",
       add = c("mean_sd",'point'),
       add.params = list(color ='black',size = 0.5), 
       point.color='blue',
       shape = 22,
       stroke=1,
       palette = "jco",
       plot_type = 'b',
       ylab = "log(D)",
       xlab = "Matrix Occupation Density",
       ylim = c(0,2.5))+
  stat_compare_means(label = "p.signif", 
                     method = "t.test",
                     ref.group = "0") +ylab(expression(log(D)))+
  my_theme   
output <- ggarrange(slopePlot,interceptPlot)
output2 <- ipmd000[[1]]
ggsave(filename = paste("./figures/diffusionCoef.pdf"),plot=output,width = 6.3, height = 3.91, unit= 'in')
ggsave(filename = paste("./figures/diffusionTime.pdf"),plot=output2,width = 6.3, height = 3.91, unit= 'in')
compare_means(intercept ~ density, ref.group = "0", data = fits,method = 't.test')
