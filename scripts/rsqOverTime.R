#########
# Displacement Squared Over Time. Questioning whether agent movment is 
# superdiffusive, diffusive or subdiffusive 
#

# Outline 
# 1 Import and Wrangling. Output -> Everything in the same dataframe 

# 2 Calculate Displacement Squared from T=0 to T=3650

# 3 Visually comparison
# 3.1 Dynamics between replicates with the same matrix conditions 
# 3.2 All Runs 

# 4 Fit Comparisons
# 4.1 Fit a Power-law to the data, grouped by each replicate  
#   Output -> fit parameters  
# 4.2 Aggregate fit parameters by matrix condition and make Boxplot 
# 4.3 Test for significance 

# 5 Do steps 2- 4 for Between T = 0 and 1000

# 6 Do steps 2- 4 for Between T = 1000 and 3650
###########################################
library(tidyverse)
library(dplyr)
library(ggplot2)
library(broom)
getwd()
# 1 Import and Wrangling. Output -> Everything in the same dataframe 
# temp = list.files(path ='./simExp/Output'  ,pattern="agent_log.........csv")
temp000 <- list.files(path ='./simExp/Output'  ,pattern="agent_log_FC100.*..csv")
temp000 <- sort(temp000)

dens000_BigList <- lapply(paste('./simExp/Output/',temp000,sep = ''), read.csv)
names(dens000_BigList) = temp000
head(dens000_BigList[[1]])
dens000_df <- bind_rows(dens000_BigList,.id = 'id') %>% 
  mutate(seed = gsub('.csv','',substring(id,17)),.keep='unused') %>% 
  select(tick,agent_id,y,seed)
dens000_df$density <- 0
# Choose starting tick 
starttick = 1.1
dens000_startpos <- dens000_df %>% filter(tick == starttick)
dens000_df <- dens000_df %>% filter(tick > starttick)

colnames(dens000_startpos)[3] = 'y*'

dens000_df <- dens000_startpos %>% select(agent_id,seed,`y*`) %>%
  right_join(dens000_df,by = c("agent_id","seed"))

diff000_df <-dens000_df %>% mutate(dy_sqrd = (y-`y*`)^2,.keep='unused') 

meandiff000_df_gbtick <- diff000_df %>% group_by(tick) %>%
  summarise(mean_dy_sqrd = mean(dy_sqrd),d_dy_sqrd = sd(dy_sqrd))

meandiff000_df_gbtickNseed <- diff000_df %>% group_by(tick,seed) %>%
  summarise(mean_dy_sqrd = mean(dy_sqrd),d_dy_sqrd = sd(dy_sqrd))

meandiff000_df_gbtick$slopeRef <- log(meandiff000_df_gbtick$tick)
meandiff000_df_gbtickNseed$slopeRef <- log10(meandiff000_df_gbtickNseed$tick)
ggplot(meandiff000_df_gbtick,aes(log(tick),log(mean_dy_sqrd)))+
  geom_line()+
  geom_line(aes(log(tick),slopeRef+3),col='red')+
  #geom_line(aes(tick,mean_dy_sqrd+d_dy_sqrd))+
  #geom_line(aes(tick,mean_dy_sqrd-d_dy_sqrd))+
  geom_smooth(formula = y ~ x,col='black',se=TRUE,method = 'lm')



ggplot(meandiff000_df_gbtickNseed,aes(log10(tick),log10(mean_dy_sqrd),col=seed))+
  geom_line(linewidth = 1)+
  geom_line(aes(log10(tick),slopeRef+1.5),col='red',linewidth=0.5,linetype = 'dashed')+
  facet_wrap(~seed)+
  geom_smooth(formula = y ~ x,col='black',se=T, method = 'lm')+
  theme_classic()

fitted_models = meandiff000_df_gbtickNseed %>% group_by(seed) %>% do(model = tidy(lm(log(mean_dy_sqrd) ~ log(tick), data = .)))

coefficients <- fitted_models %>% summarize(intercept = model$estimate[1],slope = model$estimate[2])

coefficients$density = 0

ggplot(coefficients, aes(x=density,y=slope))+
  geom_boxplot()

ggplot(coefficients, aes(x=density,y=intercept))+
  geom_boxplot()