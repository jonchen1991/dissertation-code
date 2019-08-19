#!/usr/bin/env Rscript

# Time script
time.start <- proc.time()
library(GA)
library(kernlab)

# Read in data
height0 <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/trainingset/HCdesmatrix-int0")
height1 <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/trainingset/HCdesmatrix-int1")
height2 <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/trainingset/HCdesmatrix-int2")
desmatrix.raw <- cbind(height0,height1,height2)
colnames(desmatrix.raw) <- seq(ncol(desmatrix.raw))
desmatrix <- as.matrix(desmatrix.raw)
load("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/svm.pca.features")
desmatrix <- desmatrix[,svm.pca.features]
response.raw <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/compound_class")
response <- as.matrix(response.raw)

# Variables to control GA
cost.min <- 0.01
cost.max <- 1
cost.step <- 0.01
elitism.rate <- 0.7
crossover.chance <- 0.8
mutation.chance <- 0.1
features.total <- ncol(desmatrix)
population.size <- 1000
iter.max <- 10000
run <- 100
set.seed(0)

# Create initial population and other things to input into GA
initpop <- matrix(as.double(NA),nrow=population.size,ncol=features.total)
for (count in 1:population.size){
    pop <- rep(0,features.total)
    ones <- sample(features.total,sample(features.total,1))
    pop[ones] <- 1
    initpop[count,] <- pop
}
save(initpop,file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm_initpop.RData")
load("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm_initpop.RData")
print("Initial Population Created!")
cost <- seq(cost.min,cost.max,cost.step)

# Variables to keep track of
svm.min <- list()
cross.min <- 100000
error.min <- 100000
count <- 1
svm.cost<-vector()
svm.rng<- list()

# Create model
selection <- function(string,cost){
    test.matrix <- desmatrix[,which(string==1)]
    leave.one.out <- nrow(desmatrix)
    random.cost<- sample(cost,1)
    current.rng.state<- get(".Random.seed", .GlobalEnv)
    svm.model <- ksvm(test.matrix,scaled=F,response,type="C-svc",kernel="vanilladot",C=random.cost,nu=0.2,cross=10)
    if(svm.model@cross == cross.min){
        if(svm.model@error == error.min){
            count <<- count+1
            svm.cost[count] <<- random.cost
            svm.min[[count]] <<- svm.model
            svm.rng[[count]] <<-current.rng.state
            save(svm.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.min")
            save(svm.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.rng")
        }
        if(svm.model@error < error.min){
            count <<- 1
            svm.min <<- list()
            svm.rng<<-list()
            svm.cost<<-vector()
            svm.cost[count] <<- random.cost
            svm.min[[count]] <<- svm.model
            svm.rng[[count]]<<- current.rng.state
            error.min <<- svm.model@error
            save(svm.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.min")
            save(svm.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.rng")
        }
    }
    if(svm.model@cross < cross.min){
        count <<- 1
        svm.min <<-list()
        svm.rng<<-list()
        cross.min <<- svm.model@cross
        svm.cost<<-vector()
        svm.cost[count] <<- random.cost
        svm.min[[count]] <<- svm.model
        svm.rng[[count]]<<- current.rng.state
        error.min <<- svm.model@error
        save(svm.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.min")
        save(svm.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.rng")
    }
    return(-svm.model@cross)
}

# Genetic Algorithm Implementation
GAmodel <- ga(type="binary",cost=cost,fitness=selection,nBits=features.total,popSize=population.size,suggestions=initpop,pcrossover=crossover.chance,pmutation=mutation.chance,elitism=base::max(1,round(population.size*elitism.rate)), maxiter=iter.max, run=run, keepBest=F, parallel=F,seed=0)
time.stop <- proc.time()
print(time.stop - time.start)
print("minimum x validation error")
print(cross.min)
print("minimum training error")
print(error.min)
print(summary(GAmodel))
for (i in 1:length(svm.min)){
    print(svm.min[[i]])
    print("Cost")
    print(svm.cost[i])
    print("RNG state")
    print(svm.rng[[i]])
    print("Support Vectors")
    print(svm.min[[i]]@SVindex)
    print("Features Used")
    print(as.numeric(gsub("X","",colnames(svm.min[[i]]@xmatrix[[1]]))))
    print("alpha.x")
    print(as.numeric(svm.min[[i]]@coef[[1]]%*%svm.min[[i]]@xmatrix[[1]]))
    print("intercept/b")
    print(svm.min[[i]]@b)
}
save(svm.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.min")
save(svm.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svm.rng")
