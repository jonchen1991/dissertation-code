#!/usr/bin/env Rscript

# Time script
time.start <- proc.time()
library(GA)
library(kernlab)

# Read in data
height0 <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/activity/HCdesmatrix-int0")
height1 <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/activity/HCdesmatrix-int1")
height2 <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/activity/HCdesmatrix-int2")
desmatrix.raw <- cbind(height0,height1,height2)
colnames(desmatrix.raw) <- seq(ncol(desmatrix.raw))
desmatrix <- as.matrix(desmatrix.raw)
load("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/svr.pca.features")
desmatrix <- desmatrix[,svr.pca.features]
response.raw <- read.table("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/activity.log")
response <- as.matrix(scale(response.raw)[,])

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
save(initpop,file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr_initpop.RData")
load("/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr_initpop.RData")
print("Initial Population Created!")
cost <- seq(cost.min,cost.max,cost.step)

# Variables to keep track of
svr.min <- list()
cross.min <- 100000
error.min <- 100000
count <- 1
svr.cost <- vector()
svr.rng<- list()

# Create model
selection <- function(string,cost){
    test.matrix <- desmatrix[,which(string==1)]
    leave.one.out <- nrow(desmatrix)
    random.cost<- sample(cost,1)
    current.rng.state<- get(".Random.seed", .GlobalEnv)
    svr.model <- ksvm(test.matrix,scaled=F,response,type="nu-svr",kernel="vanilladot",C=random.cost,nu=0.2,cross=10)
    if(svr.model@cross == cross.min){
        if(svr.model@error == error.min){
            count <<- count+1
            svr.cost[count] <<- random.cost
            svr.min[[count]] <<- svr.model
            svr.rng[[count]] <<-current.rng.state
            save(svr.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.min")
            save(svr.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.rng")
        }
        if(svr.model@error < error.min){
            count <<- 1
            svr.min <<- list()
            svr.rng<<-list()
            svr.cost<<-vector()
            svr.cost[count] <<- random.cost
            svr.min[[count]] <<- svr.model
            svr.rng[[count]]<<- current.rng.state
            error.min <<- svr.model@error
            save(svr.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.min")
            save(svr.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.rng")
        }
    }
    if(svr.model@cross < cross.min){
        count <<- 1
        svr.min <<-list()
        svr.rng<<-list()
        cross.min <<- svr.model@cross
        svr.cost<<-vector()
        svr.cost[count] <<- random.cost
        svr.min[[count]] <<- svr.model
        svr.rng[[count]]<<- current.rng.state
        error.min <<- svr.model@error
        save(svr.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.min")
        save(svr.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.rng")
    }
    return(-svr.model@cross)
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
for (i in 1:length(svr.min)){
    print(svr.min[[i]])
    print("Cost")
    print(svr.cost[i])
    print("RNG state")
    print(svr.rng[[i]])
    print("Support Vectors")
    print(svr.min[[i]]@SVindex)
    print("Features Used")
    print(as.numeric(gsub("X","",colnames(svr.min[[i]]@xmatrix))))
    print("alpha.x")
    print(as.numeric(svr.min[[i]]@coef%*%svr.min[[i]]@xmatrix))
    print("intercept/b")
    print(svr.min[[i]]@b)
}
save(svr.min, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.min")
save(svr.rng, file="/home/visco/jjc/dissertation/PAINFREE_AID_787_v2/10fold/jjc0/svr.rng")
