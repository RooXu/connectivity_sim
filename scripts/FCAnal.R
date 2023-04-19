filePattern = "agent_log_FC000.*..csv"

temp <- list.files(path ='./simExp/Output'  ,pattern=filePattern)
temp <- sort(temp)

dens_BigList <- lapply(paste('./simExp/Output/',temp,sep = ''), read.csv)
names(dens_BigList) = temp
head(dens_BigList[[1]])

dens_df <- bind_rows(dens_BigList,.id = 'id') %>% 
  mutate(seed = gsub('.csv','',substring(id,17)),.keep='unused') %>% 
  select(tick,agent_id,y,seed)

starttick = 1.1
endtick = 3649.1

agentsInATi <- dens_df %>% filter(tick==starttick & y <= 133.33) %>% mutate(seed = seed, agent_id = agent_id, startPatch = 'A',tick=tick)

agentsInBTi <- dens_df %>% filter(tick==starttick & (y > 133.33 & y <= 266.67)) %>% mutate(seed = seed, agent_id = agent_id, startPatch = 'B',tick=tick)

agentsInCTi <- dens_df %>% filter(tick==starttick & y > 266.67) %>% mutate(seed = seed, agent_id = agent_id, startPatch = 'C',tick=tick)

agentsInCTi %>% summarise(count=n())

agentStartPatch <- bind_rows(agentsInATi,agentsInBTi,agentsInCTi)

agentsInATf <- dens_df %>% filter(tick==endtick & y <= 133.33) %>% mutate(seed = seed, agent_id = agent_id, endPatch = 'A',tick=tick)

agentsInBTf <- dens_df %>% filter(tick==endtick & (y > 133.33 & y <= 266.67)) %>% mutate(seed = seed, agent_id = agent_id, endPatch = 'B',tick=tick)

agentsInCTf <- dens_df %>% filter(tick==endtick & y > 266.67) %>% mutate(agent_id = agent_id, endPatch = 'C',tick=tick)

agentEndPatch <- bind_rows(agentsInATf,agentsInBTf,agentsInCTf)

head(agentEndPatch)

transitionTable <- left_join(agentStartPatch,agentEndPatch,by = c('agent_id','seed')) %>% select(tickStart = tick.x,
                                                                              tickEnd = tick.y,
                                                                              seed,
                                                                              agent_id,
                                                                              startPatch,
                                                                              endPatch)
ata <- transitionTable %>% group_by(seed) %>% filter(startPatch == 'A' & endPatch =='A') %>%  summarise(ata_num = n())

atb <- transitionTable %>% group_by(seed) %>% filter(startPatch == 'A' & endPatch =='B') %>%  summarise(atb_num = n())

atc <- transitionTable %>% group_by(seed) %>% filter(startPatch == 'A' & endPatch =='C') %>%  summarise(atc_num = n())

AtX_trans_counts <- bind_cols(ata,atb,atc) %>% select(seed...1,ata_num,atb_num,atc_num)

FC <- AtX_trans_counts %>%  mutate(seed = seed...1, funconn_AC = atc_num/(ata_num+atb_num+atc_num),
                                   funconn_AB = atb_num/(ata_num+atb_num+atc_num),.keep = 'unused')

ggplot(FC,aes(x=seed))+
  geom_point(aes(y=funconn_AC),col='red')+
  geom_point(aes(y=funconn_AB),col='blue')

ggplot(FC, aes(x=funconn_AC))+
  geom_histogram()


# bta %>% group_by(seed) %>% filter(startPatch == 'B' & endPatch =='A') %>%  summarise(BtA_num = n())
# 
# btb %>% group_by(seed) %>% filter(startPatch == 'B' & endPatch =='B') %>%  summarise(BtB_num = n())
# 
# btc %>% group_by(seed) %>% filter(startPatch == 'B' & endPatch =='B') %>%  summarise(BtB_num = n())
