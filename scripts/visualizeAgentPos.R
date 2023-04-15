library(dplyr)
library(ggplot2)
library(reshape2)

getwd()
data <- read.csv("output/agent_log_FC100.csv")
#worldData <- as.matrix(read.csv("ouput/perlinRasterSeed_8.csv",header=FALSE))
#colnames(worldData) <- paste(1:5)
#rownames(worldData) <- paste(1:5)
#head(worldData)

#longWorldData <- melt(worldData)
#head(longWorldData)
#colnames(longWorldData) <- c("x", "y", "value")
#head(longWorldData)
dataframe <- as.data.frame(data)
dataframe$agent_uid_rank <- as.factor(dataframe$agent_uid_rank)
head(dataframe)
ggplot(data = filter(dataframe, tick >3649 ) ,aes(x=y,y=x,col = log_cum_prob,label=agent_id))+
  #geom_tile(data = longWorldData,aes(x=x,y=y,fill=value))+
  geom_point(size=0,alpha=1.5)+
  geom_text()+
  xlim(c(0,400))+
  ylim(c(0,400))+
  scale_colour_gradient(low = "green", high = "red", na.value = NA)
  #geom_point(data = filter(dataframe,tick == 999.1) ,aes(x=x,y=y, col='green'),size=2)

ggplot(data = filter(dataframe, tick <500 ) ,aes(x=y,y=x,col = log_cum_prob,label=agent_id))+
  #geom_tile(data = longWorldData,aes(x=x,y=y,fill=value))+
  geom_point(size=1,alpha=0.05)+
  xlim(c(0,400))+
  ylim(c(0,400))+
  scale_colour_gradient(low = "green", high = "red", na.value = NA)
#geom_point(data = filter(dataframe,tick == 999.1) ,aes(x=x,y=y, col='green'),size=2)

ggplot()+
  #geom_tile(data = longWorldData,aes(x=x,y=y,fill=value))+
  geom_path(data = filter(dataframe,agent_uid_rank == 0 & agent_id ==10 &tick < 3650) ,aes(x=y,y=x, col=log_cum_prob),alpha=0.5)+
  #geom_point(data = filter(dataframe,agent_uid_rank == 0 & agent_id == 0& 0<tick &tick < 500) ,aes(x=x,y=y,col=tick),size=40,alpha=0.01)+
  xlim(c(0,400))+
  ylim(c(0,400))+
  scale_colour_gradient(low = "green", high = "red", na.value = NA)

ggplot()+
  geom_line(data = filter(group_by(dataframe,agent_id),agent_id == 0), aes(x=tick,y=log_cum_prob,color=agent_id))+
  geom_line(data = filter(group_by(dataframe,agent_id),agent_id == 1), aes(x=tick,y=log_cum_prob,color=agent_id))+
  geom_line(data = filter(group_by(dataframe,agent_id),agent_id == 2), aes(x=tick,y=log_cum_prob,color=agent_id))

averageDistance = dataframe %>% group_by(tick) %>% summarise(meanY = mean(y), std = sd(y))

averageDistance$meanY[length(averageDistance$meanY)]
averageDistance$std[length(averageDistance$meanY)]
ggplot(filter(averageDistance,tick<3640),aes(tick,meanY))+
  geom_line()+
  geom_line(aes(tick,(meanY+std)),col = 'blue',alpha = 0.3)+
  geom_line(aes(tick,(meanY-std)),col = 'blue',alpha = 0.3)

dataframe <- dataframe %>% mutate(stepsizes = ) 
