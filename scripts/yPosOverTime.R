library(dplyr)
library(ggplot2)
# Load Density = 0 Cases
FC000_1 <- read.csv("./simExp/Output/agent_log_FC000_12.csv") # Change file names as needed
FC000_1 <- as.data.frame(FC000_1)
FC000_1$seed = 2
FC000_1$RunID = 1
## Calculate Average Y 
FC000_1_AvgY <- FC000_1 %>% group_by(tick) %>% summarise(meanY = mean(y), maxY = max(y), std = sd(y), seed = seed, RunID = RunID)

# Load Density = 20 Cases
FC020_1 <- read.csv("./simExp/Output/agent_log_FC020_3.csv") # Change file names as needed
FC020_1 <- as.data.frame(FC020_1)
FC020_1$seed = 2
FC020_1$RunID = 2

FC020_1_AvgY <- FC020_1 %>% group_by(tick) %>%  summarise(meanY = mean(y), maxY = max(y), std = sd(y), seed = seed, RunID = RunID)


# Load Density = 40 Cases
FC040_1 <- read.csv("./simExp/Output/agent_log_FC040_3.csv") # Change file names as needed
FC040_1 <- as.data.frame(FC040_1)
FC040_1$seed = 2
FC040_1$RunID = 3
FC040_1_AvgY <- FC040_1 %>% group_by(tick) %>%  summarise(meanY = mean(y), maxY = max(y), std = sd(y), seed = seed, RunID = RunID)


# Load Density = 60 Cases
FC060_1 <- read.csv("./simExp/Output/agent_log_FC060_3.csv") # Change file names as needed
FC060_1 <- as.data.frame(FC060_1)
FC060_1$seed = 2
FC060_1$RunID = 4
FC060_1_AvgY <- FC060_1 %>% group_by(tick) %>%  summarise(meanY = mean(y), maxY = max(y), std = sd(y), seed = seed, RunID = RunID)


# Load Density = 80 Cases
FC080_1 <- read.csv("./simExp/Output/agent_log_FC080_3.csv") # Change file names as needed
FC080_1 <- as.data.frame(FC080_1)
FC080_1$seed = 2
FC080_1$RunID = 5
FC080_1_AvgY <- FC080_1 %>% group_by(tick) %>%  summarise(meanY = mean(y), maxY = max(y), std = sd(y), seed = seed, RunID = RunID)


# Load Density = 100 Cases
FC100_1 <- read.csv("./simExp/Output/agent_log_FC100_3.csv") # Change file names as needed
FC100_1 <- as.data.frame(FC100_1)
FC100_1$seed = 2
FC100_1$RunID = 6
FC100_1_AvgY <- FC100_1 %>% group_by(tick) %>%  summarise(meanY = mean(y), maxY = max(y), std = sd(y), seed = seed, RunID = RunID)


all_AvgY <- rbind(FC000_1_AvgY,FC020_1_AvgY,FC040_1_AvgY,FC060_1_AvgY,FC080_1_AvgY,FC100_1_AvgY) 
all_AvgY$RunID <- as.factor(all_AvgY$RunID)

ggplot(all_AvgY,aes(x = tick, y = meanY, col = RunID ))+
  geom_line()+
  ylim(0,400)+
  scale_color_discrete(name = "Matrix Density", labels = c('0%', '20%', '40%', '60%','80%','100%'))+
  theme_classic()

ggplot(all_AvgY,aes(x = tick, y = maxY, col = RunID ))+
  geom_line()+
  ylim(0,400)+
  scale_color_discrete(name = "Matrix Density", labels = c('0%', '20%', '40%', '60%','80%','100%'))+
  theme_classic()

ggplot(all_AvgY,aes(x = tick, y = std, col = RunID ))+
  geom_line()+
  ylim(0,200)+
  scale_color_discrete(name = "Maxtrix Density", labels = c('0%', '20%', '40%', '60%','80%','100%'))+
  theme_classic()

