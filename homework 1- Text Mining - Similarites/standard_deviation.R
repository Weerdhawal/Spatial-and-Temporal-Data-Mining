


data_for_sd<-as.matrix(fsb)
sd_euc<-vector()
sd_cos<-vector()
sd_jac<-vector()
x_dp<-c(2,10,50,100,500,1000)
j<-0
for(i in x_dp)
{

sel_data<- data_for_sd[1000:2000,1:i]


euc_dist_temp<-dist(sel_data,method = "euclidean")
cos_dist_temp<-dist(sel_data,method = "cosine")
jac_dist_temp<-dist(sel_data,method = "jaccard")
eudd<-sd(euc_dist_temp,na.rm = FALSE)
sd_euc<-c(sd_euc,eudd)
coss<-sd(cos_dist_temp,na.rm = FALSE)
sd_cos<-c(sd_cos,coss)
jacc<-sd(jac_dist_temp,na.rm = FALSE)
sd_jac<-c(sd_jac,jacc)
j<-j+1
}

plot(x_dp,sd_euc,col="yellow",type="l",ylim=c(-2,20),xlab = "features",ylab="standard deviation")
lines(x_dp,sd_cos,col="red",ylim=c(-2,20))
points(x_dp,sd_jac,col="green",ylim=c(-2,20))





