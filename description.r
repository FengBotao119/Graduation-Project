library(tidyverse)

path<- 'E:/data/'
train_path<-'train_split_Depression_AVEC2017.csv'
dev_path<-'dev_split_Depression_AVEC2017.csv'
test_path <-'full_test_split.csv'


train_ros_path<-'train_ROS.csv'
#dev_ros_path <- 'dev_ROS.csv'
train <-read_csv(paste(path,train_path,sep=''))
train$PHQ8_Binary <- ifelse(train$PHQ8_Binary==1,'抑郁症患者','健康人员')
train$Gender <-ifelse(train$Gender==1,'男性','女性')


dev <-read_csv(paste(path,dev_path,sep=''))
dev$PHQ8_Binary <-ifelse(dev$PHQ8_Binary==1,'抑郁症患者','健康人员')
dev$Gender <-ifelse(dev$Gender==1,'男性','女性')

test<-read_csv(paste(path,test_path,sep=''))
test$PHQ8_Binary <-ifelse(test$PHQ_Binary==1,'抑郁症患者','健康人员')
test$Gender <-ifelse(test$Gender==1,'男性','女性')


#dev_ros<-read_csv(paste(path,dev_ros_path,sep=''))
train_ros <- read_csv(paste(path,train_ros_path,sep=''))
train_ros$Gender<- as.factor(train_ros$Gender)
train_ros$PHQ8_Binary<-as.factor(train_ros$PHQ8_Binary)
train_ros$gender <-ifelse(train_ros$Gender==1,'男性','女性')


ggplot(train)+
  geom_bar(aes(x=PHQ8_Score))+
  geom_vline(xintercept = 10,col= 'red',linetype="dashed",size=1)



ggplot(train)+
  geom_bar(aes(x=PHQ8_Score))+
  geom_vline(xintercept = 10, colour="red", linetype="dashed",size=1)+
  facet_grid(Gender~.)

ggplot(train)+
  geom_boxplot(aes(x=Gender,y=PHQ8_Score,fill=Gender),alpha= 1)+
  scale_fill_brewer(palette="Pastel1")


ggplot(train_ros)+
  geom_bar(aes(x=PHQ8_Binary,fill = Gender),col = 'black',position = position_dodge(0.95))+
  scale_fill_brewer(palette="Pastel1")

a<- ggplot(train_ros)+
  geom_bar(aes(x=PHQ8_Score))+
  geom_vline(xintercept = 10,col= 'red',linetype="dashed",size=1)+
  xlab('PHQ8分数')

b<-ggplot(train_ros)+
  geom_bar(aes(x=PHQ8_Score))+
  geom_vline(xintercept = 10, colour="red", linetype="dashed",size=1)+
  facet_grid(gender~.)+
  xlab('PHQ8分数')+
  ylab('')


ggplot(train_ros)+
  geom_boxplot(aes(x=Gender,y=PHQ8_Score,fill=Gender),alpha= 1)+
  scale_fill_brewer(palette="Pastel1")




ggplot(train)+
  geom_bar(aes(x=PHQ8_Binary,fill = factor(train$Gender,levels = c('女性','男性'))),col = 'black',position = position_dodge(0.95))+
  scale_fill_brewer(palette="Pastel1")+
  xlab('')+
  ylab('')+
  guides(fill= guide_legend( title = "性别"))
  
ggplot(dev)+
  geom_bar(aes(x=PHQ8_Binary,fill = factor(dev$Gender,levels = c('女性','男性'))),col = 'black',position = position_dodge(0.95))+
  scale_fill_brewer(palette="Pastel1")+
  xlab('')+
  ylab('')+
  guides(fill= guide_legend( title = "性别"))


ggplot(test)+
  geom_bar(aes(x=PHQ_Binary,fill = factor(test$Gender,levels = c('女性','男性'))),col = 'black',position = position_dodge(0.95))+
  scale_fill_brewer(palette="Pastel1")+
  xlab('')+
  ylab('')+
  guides(fill= guide_legend( title = "性别"))

train$belongs <- '训练集'
dev$belongs <-'验证集'
test$belongs<-'测试集'
train[,c('PHQ8_Score','PHQ8_Binary','Gender')]
dev[,c('PHQ8_Score','PHQ8_Binary','Gender')]
test[,c('PHQ_Score','PHQ8_Binary','Gender')]
test$PHQ8_Score<- test$PHQ_Score
data = rbind(train[,c('Participant_ID','PHQ8_Score','PHQ8_Binary','Gender')],dev[,c('Participant_ID','PHQ8_Score','PHQ8_Binary','Gender')],test[,c('Participant_ID','PHQ8_Score','PHQ8_Binary','Gender')])

ggplot(data)+
  geom_bar(aes(x=PHQ8_Binary,fill = factor(data$Gender,levels = c('女性','男性'))),col = 'black',position = position_dodge(0.95))+
  scale_fill_brewer(palette="Pastel1")+
  xlab('')+
  ylab('')+
  guides(fill= guide_legend( title = "性别"))+
  facet_grid(.~factor(belongs,levels = c('训练集','验证集','测试集')))


c<-ggplot(data)+
  geom_bar(aes(x=PHQ8_Score))+
  geom_vline(xintercept = 10,col= 'red',linetype="dashed",size=1)+
  ylab('count')+
  xlab('PHQ8分数')

library(gridExtra)

b<-ggplot(data)+
  geom_boxplot(aes(x=Gender,y=PHQ8_Score,fill=factor(data$Gender,levels = c('女性','男性'))),alpha= 1)+
  guides(fill= FALSE)+
  scale_fill_brewer(palette="Pastel1")+
  xlab('性别')+
  ylab('PHQ8分数')



score_mean <-mean(data$PHQ8_Score)
score_std <-sd(data$PHQ8_Score)

a<-ggplot(data )+
  geom_point(aes(x=Participant_ID,y=PHQ8_Score),size=1.6)+
  geom_abline(slope = 0,intercept = score_mean,col='red',linetype='dashed',size=1.4)+
  geom_abline(slope = 0,intercept = score_mean+3*score_std,col='blue',linetype='dashed',size=1.4)+
  geom_abline(slope = 0,intercept =score_mean-3*score_std,col='blue',linetype='dashed',size=1.4)+
  ylim(score_mean-3*score_std,score_mean+3*score_std)+
  ylab('PHQ8分数')+
  xlab('参与者编号')
  
grid.arrange(c,b,nrow=1)


ggplot(train_ros)+
  geom_bar(aes(x=PHQ8_Score))

ggplot(train_ros)+
  geom_bar(aes(x=PHQ8_Binary,fill = factor(train_ros$Gender,levels = c('女性','男性'))),col = 'black',position = position_dodge(0.95))+
  scale_fill_brewer(palette="Pastel1")
  #xlab('')+
  #ylab('')+
  #guides(fill= guide_legend( title = "性别"))
