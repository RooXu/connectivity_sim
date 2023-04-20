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
rm(list = ls())
source("./scripts/FCAnal.R")
starttick = 1825.1 
endtick = 3649.1
fc000 <- funcConnCalc("agent_log_FC000.*..csv",0,starttick,endtick)
fc020 <- funcConnCalc("agent_log_FC020.*..csv",20,starttick,endtick)
fc040 <- funcConnCalc("agent_log_FC040.*..csv",40,starttick,endtick)
fc060 <- funcConnCalc("agent_log_FC060.*..csv",60,starttick,endtick)
fc080 <- funcConnCalc("agent_log_FC080.*..csv",80,starttick,endtick)
fc100 <- funcConnCalc("agent_log_FC100.*..csv",100,starttick,endtick)

fc_al <- bind_rows(fc000,fc020,fc040,fc060,fc080,fc100)

# Plotting
my_theme <- theme_pubr(
  base_size = 12,
  base_family = "sans",
  border = FALSE,
  margin = TRUE,
  legend = c("top", "bottom", "left", "right", "none"),
  x.text.angle = 0
)

funconn_ACplot <- ggline(fc_al,
                         x="density",
                         y="funconn_AC",
                         add = c("mean_sd",'point'),
                         add.params = list(color ='black',size = 0.5), 
                         point.color='red',
                         shape = 24,
                         stroke=1,
                         palette = "jco",
                         plot_type = 'b',
                         ylab = "Likelihood of Moving from Patch A to Patch C",
                         xlab = "Matrix Occupation Density",
                         ylim = c(0,1))+
  stat_compare_means(label = "p.signif", 
                     method = "t.test",
                     ref.group = "0") +
  my_theme

funconn_ABplot <- ggline(fc_al, 
       x="density",
       y="funconn_AB",
       add = c("mean_sd",'point'),
       add.params = list(color ='black',size = 0.5),  
       point.color='red',
       shape = 24,
       stroke=1,
       palette = "jco",
       plot_type = 'b',
       ylab = "Likelihood of Moving from Patch A to Patch B",
       xlab = "Matrix Occupation Density",
       ylim = c(0,1)) +
  stat_compare_means(label = "p.signif", 
                     method = "t.test",
                     ref.group = "0")+
  my_theme

output <- ggarrange(funconn_ACplot,funconn_ABplot)

ggsave(filename = paste("./figures/fc_ti", starttick, "tf",endtick,".pdf"),plot=output,width = 10, height = 5, unit= 'in')

compare_means(funconn_AC ~ density, ref.group = "0", data = fc_al,method = 't.test')
compare_means(funconn_AB ~ density, ref.group = "0", data = fc_al,method = 't.test')
