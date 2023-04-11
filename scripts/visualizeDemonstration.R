# visualizeDemonstration uses the data listed below to demonstrate the basic 
# movement of an agent given 4 different extreme parameters 
# Source Files: 
# - agent_log.csv; enviro_food.csv
#     - Parameters: 
      #   random.seed: 1
      #   stop.at: 2000
      #   walker.count: 1
      #   world.width: 200
      #   world.height: 200
      #   matrix.density: 0.0
      #   agent_log_file: '../output/agent_log.csv'
      #   enviro_file: '../output/enviro'
      #   time.step: 1
      #   gamma.type0: 0.8
      #   alpha.type0: 1000000
      #   beta.type0: 0.5
      #   expectation.type0: 0.5
      #   omega0: 10
      #   omega1: 0
      
# - agent_log_1.csv; enviro_1_food.csv
#     - Parameters: 
#         random.seed: 1
# #       stop.at: 2000
#         walker.count: 1
#         world.width: 200
#         world.height: 200
#         matrix.density: 0.0
#         agent_log_file: '../output/agent_log_1.csv'
#         enviro_file: '../output/enviro_1'
#         time.step: 1
#         gamma.type0: 0.8
#         alpha.type0: 0.8
#         beta.type0: 1000000
#         expectation.type0: 0.5
#         omega0: 10
#         omega1: 0

# - agent_log_2.csv; enviro_2_food.csv
#   - Paramters:
        # random.seed: 1
        # stop.at: 2000
        # walker.count: 1
        # world.width: 200
        # world.height: 200
        # matrix.density: 0.0
        # agent_log_file: '../output/agent_log_2.csv'
        # enviro_file: '../output/enviro_2'
        # time.step: 1
        # gamma.type0: 0.8
        # alpha.type0: 0.8
        # beta.type0: 0.0
        # expectation.type0: 0.5
        # omega0: 10
        # omega1: 0

# - agent_log_3.csv; enviro_3_food.csv
#   - Parameters
#       random.seed: 1
        # stop.at: 2000
        # walker.count: 1
        # world.width: 200
        # world.height: 200
        # matrix.density: 0.0
        # agent_log_file: '../output/agent_log_3.csv'
        # enviro_file: '../output/enviro_3'
        # time.step: 1
        # gamma.type0: 0.8
        # alpha.type0: 0.0
        # beta.type0: 1.0
        # expectation.type0: 0.5
        # omega0: 10
        # omega1: 0


library(dplyr)
library(ggplot2)
library(reshape2)
data <- read.csv("output/agent_log_test.csv") # Change file names as needed
layer <- as.matrix(read.csv("output/enviro_test_food.csv",header = FALSE))

colnames(layer) <- c(1:200)

layer_long = melt(layer, value.name ='Layer Value',id.vars = '')

colnames(layer_long)<- c('x','y', 'Layer_Val')

ggplot()+
  geom_raster(data= layer_long,aes(x=y,y=x,fill=Layer_Val),col='black',linewidth=0.3)+
  geom_path(data = data, aes(x=y,y=x),col='white',linewidth = 0.25,alpha=0.85)+
  geom_point(data=filter(data,tick < 2),aes(x=y,y=x),col = 'red',shape=8,size=4,stroke = 1)
