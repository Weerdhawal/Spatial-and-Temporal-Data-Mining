#libraries
library(tm)
library(class)
library(plyr)


set.seed(50000)
#Setting options
options(stringsAsFactors = FALSE)


#directories
categories <-c("alt.atheism","comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x","misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball","rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space","soc.religion.christian","talk.politics.guns","talk.politics.mideast","talk.politics.misc","talk.religion.misc")
pathname<-"I:/Masters/SPRING 18/SPATIAL AND TEMPORAL/spatial/homework 2/r project/20news-18828"


#cleantext

cleanCorpus <-function(all.corp){
 
  #remove punctuations
  temp.corp<-tm_map(all.corp,removePunctuation)
  #strip extra white spaces
  temp.corp <- tm_map(temp.corp,stripWhitespace)
  #apply Numbers from files
  temp.corp<-tm_map(temp.corp,removeNumbers)
  #lower case
  temp.corp<- tm_map(temp.corp, tolower)
  temp.corp<- tm_map(temp.corp, removeWords, stopwords(kind = "en"))
  temp.corp<-tm_map(temp.corp,removeWords,c("on","in","next to","infront of","behind","between","under","through",
                                                        " around","i","me","my","mine","you","your","yours","he","him","his","she",
                                                        "her","hers","it","its","we","us","our","ours","they","their","theirs","them","easily",
                                                        "loudly","quickly","quietly","sadly","silently","slowly","always","frequently","often","once"))
  
  return(temp.corp)
}

#build tdm

generateTDM <-function(categ,path){
  s.dir<- sprintf("%s/%s",path,categ)
  s.cor <- Corpus(DirSource(directory = s.dir,encoding = "UTF-8"),readerControl=list(reader=readPlain,language="en"))
  s.cor.clean <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.clean) 
  
  s.tdm<-removeSparseTerms(s.tdm,0.85)
  result<-list(name=categ,tdm=s.tdm)
  
}

tdm<-lapply(categories,generateTDM,path = pathname)


#attach  name

bindcategstoTDM<- function(tdm){
  s.mat <- t(data.matrix(tdm[["tdm"]]))
  s.df <- as.data.frame(s.mat, stringsAsFactors=FALSE)
  
  s.df <- cbind(s.df,rep(tdm[["name"]],nrow(s.df)))
  colnames(s.df)[ncol(s.df)]<-"mainCATEGORIES"
  return(s.df)
}

categTDM <- lapply(tdm,bindcategstoTDM)

#stack them together
tdm.stack <- do.call(rbind.fill,categTDM)
tdm.stack[is.na(tdm.stack)]<- 0


#dataset train test
#train.data <- sample(nrow(tdm.stack),ceiling(nrow(tdm.stack)*0.7))
#test.data <-(1:nrow(tdm.stack))[-train.data]


#model- Knn
tdm.categ <- tdm.stack[,"mainCATEGORIES"]
tdm.stack.nl <- tdm.stack[,!colnames(tdm.stack) %in% "mainCATEGORIES"]

knn.pred<- knn(tdm.stack.nl[train.data, ],tdm.stack.nl[test.data,],tdm.categ[train.data])
 #accuracy
conf.matrix <- table("Prediction"= knn.pred,Actual =tdm.categ[test.data])

(accuracy<- sum(diag(conf.matrix))/length(test.data)*100)
