# connectivityAnalysis uses the data listed below to demonstrate the evolution 
# of agent spatial distribution w.r.t x axis. Methods include #1 plotting the 
# aggregate Y position over time. #2 The scrolling density at several points in 
# time. #3 the probability of moving from the starting patch to another patch.

# For 0% Density Case
  # random.seed: 1
  # stop.at: 3650
  # walker.count: 64
  # world.width: 400
  # world.height: 400
  # matrix.density: 1.0
  # agent_log_file: '../output/agent_log_test_1.csv'
  # enviro_file: '../output/enviro_test_food'
  # time.step: 12
  # gamma.type0: 1.84
  # alpha.type0: 4.83
  # beta.type0: 0.01
  # expectation.type0: 0.3
  # omega0: 5
  # omega1: -5

# For 100% Density Case
  # random.seed: 1
  # stop.at: 3650
  # walker.count: 64
  # world.width: 400
  # world.height: 400
  # matrix.density: 1.0
  # agent_log_file: '../output/agent_log_FC100.csv'
  # enviro_file: '../output/enviro_test_FC100'
  # time.step: 12
  # gamma.type0: 1.84
  # alpha.type0: 4.83
  # beta.type0: 0.01
  # expectation.type0: 0.3
  # omega0: 5
  # omega1: -5

# Average Y Position Over Time 
data <- read.csv("./simExp/Output/agent_log_FC060_1.csv") # Change file names as needed
data <- as.data.frame(data)
averageYDistance <- data %>% group_by(tick) %>% summarise(meanY = mean(y), std = sd(y))
averageYDistance$meanY[length(averageYDistance$meanY)]
averageYDistance$std[length(averageYDistance$meanY)]
ggplot(filter(averageYDistance,tick<3650),aes(tick,meanY))+
  geom_line()+
  geom_line(aes(tick,(meanY+std)),col = 'blue',alpha = 0.3)+
  geom_line(aes(tick,(meanY-std)),col = 'blue',alpha = 0.3)+
  ylim(0,400)
# Distribution along Y axis at several points in time 
yDistrSnapShotT0 <- data %>% filter(tick == 1.1) %>% select(y)
ggplot(yDistrSnapShotT0)+
  stat_density(aes(y))+
  xlim(0,400)

yDistrSnapShotTx1 <- data %>% filter(tick == 500.1) %>% select(y)
ggplot(yDistrSnapShotTx1)+
  stat_density(aes(y))+
  xlim(0,400)

yDistrSnapShotTx2 <- data %>% filter(tick == 1000.1) %>% select(y)
ggplot(yDistrSnapShotTx2)+
  geom_density(aes(y))+
  xlim(0,400)

yDistrSnapShotTF <-  data %>% filter(tick == 3649.1) %>% select(y)
ggplot(yDistrSnapShotTF)+
  geom_density(aes(y))+
  xlim(0,400)

ggplot()+
  stat_density(data = yDistrSnapShotT0,aes(y,fill = 'T=0'),alpha = 1.0)+
  stat_density(data = yDistrSnapShotTx1,aes(y,fill = 'T=500'),alpha = 0.75)+
  stat_density(data = yDistrSnapShotTx2,aes(y,fill = 'T=1000'),alpha = 0.60)+
  stat_density(data = yDistrSnapShotTF,aes(y,fill = 'T=3549'),alpha = 0.5)+
  xlim(0,400)

# Functional Connectivity measured as interested patch transitions over
# total patch transitions
countinAT0 <- data %>% filter(tick == 1.1 & y < 400/3) %>% summarise(count = n())
countinATF <- data %>% filter(tick == 3649.1 & y < 400/3) %>% summarise(count = n())
countinBTF <- data %>% filter(tick == 3649.1 & y > 400/3 & y < 400*2/3) %>% summarise(count = n())
countinCTF <- data %>% filter(tick == 3649.1 & y > 400*2/3) %>% summarise(count = n())

FC = countinATF/(countinATF+countinBTF+countinCTF)
