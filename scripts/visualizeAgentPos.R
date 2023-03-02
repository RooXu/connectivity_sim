library(dplyr)
library(ggplot2)
library(reshape2)
getwd()
data <- read.csv("output/agent_log_111.csv")
worldData <- as.matrix(read.csv("output/perlinRasterSeed_8.csv",header=FALSE))
colnames(worldData) <- paste(1:1000)
rownames(worldData) <- paste(1:1000)
head(worldData)

longWorldData <- melt(worldData)
head(longWorldData)
colnames(longWorldData) <- c("x", "y", "value")
head(longWorldData)
dataframe <- as.data.frame(data)
head(dataframe)
ggplot()+
  geom_tile(data = longWorldData,aes(x=x,y=y,fill=value))+
  geom_point(data = filter(dataframe,tick == 1.1) ,aes(x=x,y=y, col='red'),size=0.1)+
  geom_point(data = filter(dataframe,tick == 999.1) ,aes(x=x,y=y, col='green'),size=0.1)
ggplot()+
  geom_tile(data = longWorldData,aes(x=x,y=y,fill=value))+
  geom_path(data = filter(dataframe,agent_id == 70) ,aes(x=x,y=y, col='red'))+
  xlim(c(0,1000))+
  ylim(c(0,1000))
