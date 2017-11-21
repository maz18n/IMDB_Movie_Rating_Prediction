library(ggplot2)
library(ggcorrplot)
library(MASS)
library(corrgram)
library(dplyr)
library(PerformanceAnalytics)
library(corrplot)
library(caret)
library(plotly)
library(car)
library(gbm)
library(boot)
library(VIM)
require(randomForest)
library(rfUtilities)
library(tm)
library(googleVis)
library(dplyr)
library(Metrics)
movie <- read.csv("C:/Users/maz/Desktop/movie_rating_prediction-master/movie_metadata.csv", na.strings = "")
summary(movie)

length(unique(movie$director_name))#more than 2000 categories.. can't be used as variable
length(unique(movie$actor_1_name))#more than 2000 categories.. can't be used as variable
length(unique(movie$actor_3_name))#more than 2000 categories.. can't be used as variable
length(unique(movie$plot_keywords))#more than 2000 categories.. can't be used as variable
length(unique(movie$genres))##text mining to genres to get document term matrix and adding new variables
length(unique(movie$color)) # can be used for modelling as categorica varible
length(unique(movie$actor_2_name)) ##more than 2000 categories.. can't be used as variable
length(unique(movie$content_rating)) # only 19 categories can be used for modelling as categorical variable
length(unique(movie$language)) # only 48 categories, therefor can be used for modelling
length(unique(movie$country)) # only 66, therefor countries can be used for modelling
length(unique(movie$aspect_ratio))# only 23, therefor can be used for modelling


#Data Exploration
summary(movie) #many missing values in variables like gross
##missing value plot

## Matrix plot. Red for missing values, Darker values are high values.
matrixplot(movie, interactive=T, sortby="imdb_score")
#Knn missing value imputation using VIM package
movie = kNN(movie)
movie = movie[,-c(29:56)]
summary(movie)#no missing values
# top actors

colnames(movie)
m1 = movie %>% select(actor_1_name, actor_1_facebook_likes) %>% 
  group_by(actor_1_name) %>% summarize(appear.count=n())

m2 = left_join(movie, m1, by="actor_1_name")
m3 = m2 %>% select(actor_1_name, actor_1_facebook_likes, appear.count) %>%
  distinct %>% arrange(desc(appear.count))

hist(m3$appear.count, breaks=30)

Bubble <- gvisBubbleChart(m3, idvar="actor_1_name", 
                          xvar="appear.count", yvar="actor_1_facebook_likes",
                          sizevar="appear.count",
                          #colorvar="title_year",
                          options=list(
                            #hAxis='{minValue:75, maxValue:125}',
                            width=1000, height=800
                          )
)
plot(Bubble)


# top actors 2

m1 = movie %>% select(actor_2_name, actor_2_facebook_likes) %>% 
  group_by(actor_2_name) %>% summarize(appear.count=n())

m2 = left_join(movie, m1, by="actor_2_name")
m3 = m2 %>% select(actor_2_name, actor_2_facebook_likes, appear.count) %>%
  distinct %>% arrange(desc(appear.count))

hist(m3$appear.count, breaks=30)

Bubble <- gvisBubbleChart(m3, idvar="actor_2_name", 
                          xvar="appear.count", yvar="actor_2_facebook_likes",
                          sizevar="appear.count",
                          #colorvar="title_year",
                          options=list(
                            #hAxis='{minValue:75, maxValue:125}',
                            width=1000, height=800
                          )
)
plot(Bubble)

# top director
movie$director_facebook_likes
m1 = movie %>% select(director_name, director_facebook_likes) %>% 
  group_by(director_name) %>% summarize(appear.count=n())

m2 = left_join(movie, m1, by="director_name")
m3 = m2 %>% select(director_name, director_facebook_likes, appear.count) %>%
  distinct %>% arrange(desc(appear.count))

hist(m3$appear.count, breaks=30)

Bubble <- gvisBubbleChart(m3, idvar="director_name", 
                          xvar="appear.count", yvar="director_facebook_likes",
                          sizevar="appear.count",
                          #colorvar="title_year",
                          options=list(
                            #hAxis='{minValue:75, maxValue:125}',
                            width=1000, height=800
                          )
)
plot(Bubble)
#subsetting only numeric varibles for correlation plot
movie_numerics=movie[,-c(1,2,7,10,11,12,15,17,18,20,21,22,24,27)]
dim(movie_numerics)
####TEXT MINING OON GENRES AND PLOT KEYWORD TO CREATE NEW CATEGORICAL VARIABLES
# text mining of genres to create document term matrix
##genres DTM creation
# change this file location to suit your machine
genres= as.character(movie$genres)
genres = gsub("[[:punct:]]"," ", genres) # REMOVED punctuations
genres = gsub("Sci Fi","Sci-Fi ", genres) # reinserting hyphen in sci fi
genres_corpus <- VCorpus(VectorSource(genres)) #corpus creation of geners
genres_dtm <-DocumentTermMatrix(genres_corpus) 
dim(as.matrix(genres_dtm))#28 genres vaiables created
genres_df= data.frame(as.matrix(genres_dtm))

##text mining of plotkeywords
plotkey= as.character(movie$plot_keywords)
plotkey = gsub("[[:punct:]]"," ", plotkey)
plotkey = gsub("war","warplotkey ", plotkey)
plotkey = gsub("film","filmplotkey ", plotkey)
plotkey_corpus <- VCorpus(VectorSource(plotkey))
plotkey_corpus <- tm_map(plotkey_corpus, removeNumbers)
plotkey_corpus <- tm_map(plotkey_corpus, removeWords, stopwords("english")) # this stopword file is at C:\Users\[username]\Documents\R\win-library\2.13\tm\stopwords 
plotkey_corpus <- tm_map(plotkey_corpus, stemDocument)
plotkey_corpus_dtm <-DocumentTermMatrix(plotkey_corpus) 
plotkey_corpus_dtm <- removeSparseTerms(plotkey_corpus_dtm, 0.99) # parsity set at 99%
plotkeyword_df= data.frame(as.matrix(plotkey_corpus_dtm))
###COMBINING GENRES AND PLOT KEYWORDS TO MAKE NEW DATA FRAME AND REMOVE HIGHLY CORRELATED
genresplot=cbind(genres_df,plotkeyword_df)

# All numeric variables plus other selected categorical variables like, language, content rating, color
movie_new= cbind(movie_numerics,genresplot)
correlations <- cor(movie_new)
corrplot(correlations, order = "hclust")
highCorr <- findCorrelation(correlations, cutoff = .95)
length(highCorr)
movie_new<- movie_new[, -highCorr]
dim((movie_new))

##final data set creation combinig by cbind
movie_new$color = movie$color
movie_new$language=movie$language
movie_new$content_rating= movie$content_rating
#movie_new$country= movie$country . Country is higly correlated with language, hence is producing multicollinearity
movie_new$aspect_ratio = movie$aspect_ratio

dim(movie_new) # 139 variables in the final data set
##important categorical variables visualization for dependancy analysis

plot(density(movie_new$imdb_score))# normally distributed
abline(v=mean(movie$imdb_score), lty=2)

plot_ly(movie, x = ~title_year, y = ~imdb_score,  
        type = "box")%>%  layout(boxmode = "group")

plot_ly(movie, x = ~color, y = ~imdb_score,  
        type = "box")%>%  layout(boxmode = "group")##dependent on color/Black and weight. IMDB score varies

reordered_genres = with(movie, reorder(genres, -imdb_score, median))
plot_ly(movie, x = ~genres, y = ~imdb_score,  
        type = "box")%>%  layout(boxmode = "group")##dependent on genres. IMDB score varies

reordered_country = with(movie, reorder(country, -imdb_score, median))
plot_ly(movie, x = ~reordered_country, y = ~imdb_score,  
        type = "box")%>%  layout(boxmode = "group")##dependent on country . IMDB score varies

reordered_content_rating = with(movie, reorder(content_rating, -imdb_score, median))
plot_ly(movie, x = ~reordered_content_rating, y = ~imdb_score,  
        type = "box")%>%  layout(boxmode = "group")##dependent on . IMDB score varies
reordered_aspect_ratio = with(movie, reorder(aspect_ratio, -imdb_score, median))
plot_ly(movie, x = ~aspect_ratio, y = ~imdb_score,  
        type = "box")%>%  layout(boxmode = "group")##dependent on aspect_ratio . IMDB score varies



#linear regression
reg1 = lm(imdb_score~., data =movie_new)
AIC(reg1)
BIC(reg1)
summary(reg1) ##R square of 0.4728
rmse(movie_new$imdb_score, reg1$fitted.values) #rmse 0.817

##Regression Diagnostics with plots
opar= par()
par(mfrow=c(2,2))
plot(reg1) 
par(opar)
##checking multicollinearity
vif(reg1)   
##checking for influential observation by influence plot and cooks distance
influencePlot(reg1)

d=cooks.distance(reg1)
cutoff= 4/nrow(movie_new)
length(d[d>cutoff])
##Weigthed  least square regression to balance the influenctial observation

w= ifelse(d<cutoff,1,cutoff/d) # weight initialised
reg2 = lm(imdb_score~., data =movie_new,weights = w )

summary(reg2)

opar= par()
par(mfrow=c(2,2))
plot(reg2) 
par(opar)

rmse(movie_new$imdb_score, reg2$fitted.values)  ## r square increased after wieghted least square regression
# but rmse decreased
####
ncvTest(reg1)
##pvalue is not significant, therefor error is hetroscadasitic as per ncv test. Regression structure needs change

###Check for interaction terms
##since number of vairables is more than 100, possipple number f interaction is 2^100. 
# smaller set of variables are taken to identify is there any significatn interaction terms
movie_new1 = cbind(movie_numerics, movie[,c("color", "content_rating", "language", "aspect_ratio")])
reg_step= lm(imdb_score~., data =movie_new1)
AIC(reg_step)
reg_step1= lm(imdb_score~.+ budget:language++ duration:content_rating, data =movie_new1)
AIC(reg_step1)

#res = step(reg_step,~.^2) 
#res$anova


##Vif greater than 4 for few variables..they are removed from modelling
reg3 = lm(imdb_score~. -num_voted_users -new -language-content_rating , data =movie_new)

summary(reg3)  ##RSE increased

opar= par()
par(mfrow=c(2,2))
plot(reg3) 
par(opar)

vif(reg3)


######Regression structure 


boxCox(reg1, family="yjPower", plotit = TRUE) # lambda = 2 thus y transformation requried
#lambda  = 2, hence y should be powered
reg3 =lm(I(imdb_score)^2~., data =movie_new)  
summary(reg3)
opar= par()
par(mfrow=c(2,2))
plot(reg3) 
par(opar)

#After box cox regression model Rsquare  and regression is improved. But the RMSE is increased. Thus  transformation is not required

## Lots of vairables has zeros, hence box tidwell transforamtion can't be done

##Ridge Regression 
library(glmnet)
movie_new= na.omit(movie_new)
dim(movie_new)
colnames(movie_new)
movie_new[,1]
x=model.matrix(imdb_score~., data =movie_new )[,-12]
dim(x)
y=movie_new$imdb_score
grid =10^seq(10,-3, length =100)
set.seed (1)
train=sample (nrow(x), 0.8*nrow(x))
test=(- train )
y.test=y[test]
y.train= y[train]

cv.out =cv.glmnet(x[train,],y[train],lambda= grid, alpha =0, nfolds = 10)
ncol(x)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam
ridge.pred=predict(cv.out ,s=bestlam ,newx=x[test,])
mean((ridge.pred -y.test)^2)##rmse is 0.72

##refit ridge on whole data
out=glmnet (x,y,alpha =0)
predict (out ,type="coefficients",s=bestlam )[1:ncol(x),]

###LASSO Regression
lasso.mod =glmnet (x[train ,],y[train],alpha =1, lambda =grid)
plot(lasso.mod)
set.seed (1)
cv.out =cv.glmnet (x[train ,],y[train],lambda =grid,alpha =1, nfolds = 10)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam
lasso.pred=predict (lasso.mod ,s=bestlam ,newx=x[test,])
mean((lasso.pred -y.test)^2)###RMSE of 0.70
lasso.coef  <- predict(lasso.mod, type = 'coefficients', s = bestlam)[1:ncol(x),]
lasso.coef
out=glmnet (x,y,alpha =1, lambda =grid)
lasso.coef=predict(out,type ="coefficients",s=bestlam)[1:ncol(x),]



##PCR regression

library (pls)
set.seed (2)
pcr.fit=pcr(imdb_score~., data =movie_new,validation ="CV")
summary(pcr.fit) ##pcr cv error is  greater than 0.85 

#random forest
library(MASS)
library(randomForest)
require(randomForest)
library(rfUtilities)
dim(movie_new)

fit <- randomForest(imdb_score~.,mtry=12, data=movie_new,importance =TRUE)  


print(fit) # view results 
importance(fit) # importance of each predictor
##MSE = 0.57
#thus RMSE =0.755


### cross validation of Random forest

# Load Dataset
dim(movie_new)
x <- subset(movie_new, select= - imdb_score)
y <- subset(movie_new, select= imdb_score)
control <- trainControl(method="cv", number=4)
seed <- 7
metric <- "RMSE"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(imdb_score~., data=movie_new, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)
rf_default$coefnames

####Gradient Boosted Model using XgBoost package

library(caret)
library(plyr)
library(xgboost)
library(Metrics)
# Create custom summary function in proper format for caret
custom_summary = function(data, lev = NULL, model = NULL){
  out = rmse(data[, "obs"], data[, "pred"])
  names(out) = c("rmse")
  out
}
# Create control object
control = trainControl(method = "cv",  # Use cross validation
                       number = 5,     # 5-folds
                       summaryFunction = custom_summary                      
)

# Create grid of tuning parameters
grid = expand.grid(nrounds=c(500), # Test 4 values for boosting rounds
                   max_depth= c(6,8),           # Test 2 values for tree depth
                   eta=c( .05),      # Test 3 values for learning rate
                   gamma= c(0.1,0.2), 
                   colsample_bytree = c(0.8), 
                   min_child_weight = c(1),
                   subsample= c(1))

xgb_tree_model =  train(imdb_score~.,      # Predict SalePrice using all features
                        data=movie_new,
                        method="xgbTree",
                        trControl=control, 
                        nthreads = 4,
                        tuneGrid=grid, 
                        metric="rmse",     # Use custom performance metric
                        maximize = FALSE)   # Minimize the metric

xgb_tree_model$results
xgb_tree_model$bestTune# best model rmse of cross validation is 0.72 Xgboost package

varImp(xgb_tree_model)



######conclusion 

#best model is lasso regression with crossvalidation rmse of 0.705 with rsquare of
#followed by ridge with crossvalidation rmse 0.715
#ollowed by Gradient boosted model with cross validationRMSE pf 0.73
#pcr rmse 0.755
#followed by random forest cross validation rmse of 0.78
#followed by Linear regression 0.71




#only 20 most important variables shown (out of 198) by X


#num_voted_users         100.000
#duration                 38.183
#drama                    32.713
#budget                   26.632
#num_user_for_reviews     18.321
#num_critic_for_reviews   16.023
#movie_facebook_likes     15.944
#gross                    15.338
#documentary              11.834
#actor_3_facebook_likes   10.185
#director_facebook_likes   9.088
#actor_1_facebook_likes    8.132
#actor_2_facebook_likes    7.457
#languageEnglish           4.790
#horror                    4.475
#content_ratingPG-13       4.333
#facenumber_in_poster      3.608
#aspect_ratio              3.497
#action                    3.084
#thriller                  2.836
 









