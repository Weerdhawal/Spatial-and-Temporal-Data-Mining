#install read text adn quanteda


library(ggplot2)
library(reshape2)
require(tm)
require(SnowballC)
require(reshape2)
require(ggplot2)
library(tm)
library(gplots)
library(RColorBrewer)
library(wordcloud)
library(proxy)

#load the corpus
all <- Corpus(DirSource("I:\\Masters\\SPRING 18\\SPATIAL AND TEMPORAL\\spatial\\homework1\\20news-18828\\20news-18828\\",
                        encoding="UTF-8",recursive=TRUE),readerControl=list(reader=readPlain,language="en"))

#check values
all[[1]]

#lower case
all<- tm_map(all, tolower)
#remove punctuations
all<-tm_map(all,removePunctuation)
#strip extra white spaces
all <- tm_map(all,stripWhitespace)
#apply Numbers from files
all<-tm_map(all,removeNumbers)

#now remove the standard list of stopwords, like you've already worked out
all.nostopwords <- tm_map(all, removeWords, stopwords(kind = "en"))
#remove words specified in the document
all.nostopwords<-tm_map(all.nostopwords,removeWords,c("on","in","next to","infront of","behind","between","under","through",
                                                      " around","i","me","my","mine","you","your","yours","he","him","his","she",
                                                      "her","hers","it","its","we","us","our","ours","they","their","theirs","them","easily",
                                                      "loudly","quickly","quietly","sadly","silently","slowly","always","frequently","often","once"))

#making TF matrix
tdm<-TermDocumentMatrix(all.nostopwords,control =list(weighting= weightTf,normalize = TRUE))

#REMOVE SPARSE TERMS AND CONVERT TO MATRIX
final_tdm<- removeSparseTerms(tdm, sparse = 0.99)
final_matrix<- as.matrix(final_tdm)
#TRANSPOSE MATRIX FOR ROWS=ARTICLE , COLUMN=TERM i.e to get Document term matrix
fsb<-t(final_matrix)
#FEATURE SELECTION
#SORT VALUES to get top 100 words
fsb_s<-sort(colSums(fsb),decreasing = TRUE)
fsb_d<-data.frame(word=names(fsb_s),freq=fsb_s)
top100<-head(fsb_d,100)
#create word cloud 
wordcloud(words=names(fsb_s),freq=fsb_s,min.freq=1000,random.order=F)
colna<-names(fsb_s)


#
colna<-findFreqTerms(final_tdm, lowfreq = 2284,highfreq = 20334)
a<-data.frame(fsb)
top100_sel<-a[,colna] 
#top100_sel1<-as.matrix(top100_sel)
top100_sel1<-as.matrix(fsb)
top100_sel1<- top100_sel1[1000:2000,]

#histogram
barplot(top100[1:100,]$freq, las = 2, names.arg = top100[1:100,]$word,col ="lightblue", main ="Most Frequent Words",ylab = "Word frequencies")


#cALCULATING SIMILARITIES
euc_dist<-dist(top100_sel1,method = "euclidean")
melted_eud_d<-melt(as.matrix(euc_dist))
ggplot(data = melted_eud_d, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+ scale_fill_gradient(low = "yellow", high = "red")

cos_dist<-dist(top100_sel1,method = "cosine")
melted_cos_d<-melt(as.matrix(cos_dist))
ggplot(data = melted_cos_d, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+ scale_fill_gradient(low = "yellow", high = "red")

jac_dist<-dist(top100_sel1,method = "jaccard")
melted_jac_d<-melt(as.matrix(jac_dist))
ggplot(data = melted_jac_d, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+ scale_fill_gradient(low = "yellow", high = "red")

#setp 7

#similarities
cos_euc<-cor(euc_dist,cos_dist,method = "pearson")
euc_jac<-cor(jac_dist,euc_dist,method = "pearson")
jac_cos<-cor(cos_dist,jac_dist,method = "pearson")

#linear regression
vec_cos<-as.vector(head(cos_dist,500))
vec_euc<-as.vector(head(euc_dist,500))
vec_jac<-as.vector(head(jac_dist,500))
df_top100<-data.frame(head(top100_sel1,500))
lr_cos_euc<-lm(vec_cos~vec_euc,df_top100)
lr_euc_jac<-lm(vec_euc~vec_jac,df_top100)
lr_jac_cos<-lm(vec_jac~vec_cos,df_top100)


#plot for Simialiteries 
scatter.smooth(x=vec_euc,y=vec_cos,pch=21,col="blue",lpars =list(col = "red", lwd = 3, lty = 3),xlim=c(0,20),xlab="Euclidean",ylab="Cosine")

scatter.smooth(x=vec_cos,y=vec_jac,pch=21,col="blue",lpars =list(col = "red", lwd = 3, lty = 3),xlim=c(0,1),xlab="Cosine",ylab="Jaccard")

scatter.smooth(x=vec_jac,y=vec_euc,pch=21,col="blue",lpars =list(col = "red", lwd = 3, lty = 3),xlim=c(0,1),ylim = c(0,30),xlab="Jaccard",ylab="Euclidean")






#step 9
trial_melted_cos <- melted_cos_d
tail(trial_melted_cos[order(trial_melted_cos$value),],10)
trial_melted_jac <- melted_jac_d
tail(trial_melted_jac[order(trial_melted_jac$value),],10)
trial_melted_euc <- melted_eud_d
tail(trial_melted_euc[order(trial_melted_euc$value),],10)

#convert to csv
write.csv(top100_sel1,"top100Features.csv")

fsb<-read.csv("input.csv")







