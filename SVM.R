# Necessary libraries
library(caret)
library(XLConnect)  
library(corrplot)
library(caret)
library(doParallel)
library(e1071)
registerDoParallel(cores = 4)

# Set seed for reproducibility and also set working directory
set.seed(1)
wk = loadWorkbook("CTG.xls") 
data = readWorksheet(wk, sheet="Raw Data")

# First 3 columns are unnecessary: filenames and date
data = data[,-1:-3]

# How many observations in the column are NA 
na_columns_rate = sapply(data, FUN = function(x) sum(is.na(x))/length(x))

# Remove almost fully NA columns
data = data[,na_columns_rate < 0.9]

# Remove NA rows. There are 3 of them
data = data[complete.cases(data),]

# Make all the columns numeric
original_data_types = sapply(data, typeof)
data <- data.frame(lapply(data, FUN = function(x) as.numeric(sub(",", ".", x, fixed = TRUE))))

# Treat all columns with less than 5 unique values as factors
unique_values_count = sapply(data, FUN = function(x) length(unique(x)))
data[,unique_values_count<5] = data.frame(lapply(data[,unique_values_count<5], FUN = function(x) as.factor(as.character(x))))

# Treat CLASS as a factor as well
data$CLASS = as.factor(data$CLASS)

# Names for levels
levels(data$NSP) = c("Normal", "Suspect", "Pathologic")

# Finally, remove columns with near-zero variance
data = data[,-nearZeroVar(data, freqCut = 300/1)]

# Data for visualization
raw_data <- data.frame(data)

# Remove binary decision classes [cofounding variables. see dataset description]
data <- data[-24:-33]

# shuffle and split for training and testing
data <- data[sample(nrow(data)),]

# NSP classification
trainIndex <- sample(1:nrow(data), trunc(length(1:nrow(data))/5))
testing <- data[trainIndex, ]
training <- data[-trainIndex, ]

# seeds for parallel workers
seeds <- vector(mode = "list", length = 6) # length is = (n_repeats*nresampling)+1
for(i in 1:5) seeds[[i]]<- sample.int(n=1000, 22) # ...the number of tuning parameter...
seeds[[6]]<-sample.int(1000, 1) # for the last model

# seperate data sets
nsp_raw_training <- training[,-(length(training)-1)]
nsp_raw_testing <- testing[,-(length(testing)-1)]

#### correlation between attributes ###

### correlation filtering ###
scaledData <- scale(nsp_raw_training[,1:22], center = TRUE, scale = TRUE)
corMatrix <- cor(scaledData)

#Plot variable importance
png(filename = "attrCorrelation.png", width = 1000, height = 1000)
corrplot(corMatrix, order="hclust")
dev.off()

# get highly correlated attributes (>95%)
highCorAttrib <- findCorrelation(corMatrix, cutoff=0.95)
print("Remove predictors with >95% correlation:")
print(sort(highCorAttrib, decreasing = TRUE))

# remove attributes with high correlation
nsp_training <- nsp_raw_training[,-highCorAttrib]
nsp_testing <- nsp_raw_testing[,-highCorAttrib]

### RFE filtering ###
# feature importance
nspFeatures <- rfe(nsp_training[,1:(length(nsp_training)-1)], 
                   nsp_training$NSP, 
                   size=c(1:(length(nsp_training)-1)),
                   rfeControl=rfeControl(functions=rfFuncs, method="repeatedcv", number = 5, seeds = seeds)
)

l <- predictors(nspFeatures)
l[length(l)+1] = "NSP"

nsp_training <- nsp_training[,l]
nsp_testing <- nsp_testing[,l]
t1 <- proc.time()
fitControl <- trainControl(method="repeatedCV",number = 5,repeats = 1,index = createMultiFolds(nsp_training$NSP, k=5, times=1))
#Define Equation for Models

nsp_svm_model <- train(NSP~.,data = nsp_training,method = "svmLinear",preProc = c("center","scale"),trControl = fitControl,tuneGrid = expand.grid(C= 2^c(0:5)))
t2 <- proc.time()
nsp_svm_predict <- predict(nsp_svm_model, nsp_testing[,-(length(nsp_testing))])
nsp_svm_verification <- confusionMatrix(nsp_testing$NSP, nsp_svm_predict)
print((t2-t1)[2])
print(nsp_svm_verification$overall[1])