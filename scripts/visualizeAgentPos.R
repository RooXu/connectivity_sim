library(dplyr)
library(ggplot2)
library(reshape2)

getwd()
data <- read.csv("output/agent_log_323.csv")
#worldData <- as.matrix(read.csv("output/perlinRasterSeed_8.csv",header=FALSE))
#colnames(worldData) <- paste(1:5)
#rownames(worldData) <- paste(1:5)
#head(worldData)

#longWorldData <- melt(worldData)
#head(longWorldData)
#colnames(longWorldData) <- c("x", "y", "value")
#head(longWorldData)
dataframe <- as.data.frame(data)
head(dataframe)
ggplot()+
  #geom_tile(data = longWorldData,aes(x=x,y=y,fill=value))+
  geom_point(data = filter(dataframe) ,aes(x=y,y=x,col = agent_uid_rank),size=1)+
  xlim(c(0,200))+
  ylim(c(0,200))
  #geom_point(data = filter(dataframe,tick == 999.1) ,aes(x=x,y=y, col='green'),size=2)

ggplot()+
  #geom_tile(data = longWorldData,aes(x=x,y=y,fill=value))+
  geom_path(data = filter(dataframe,agent_uid_rank == 3 & agent_id == 0 &tick < 1000) ,aes(x=y,y=x, col=tick))+
  #geom_point(data = filter(dataframe,agent_uid_rank == 0 & agent_id == 0& 0<tick &tick < 500) ,aes(x=x,y=y,col=tick),size=40,alpha=0.01)+
  xlim(c(0,200))+
  ylim(c(0,200))
dataframe <- dataframe %>% mutate(stepsizes = ) 
