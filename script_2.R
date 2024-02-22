# Machine Learning Project
# 01/2024
# LEBRETON Louis

# Dogs data
# Multiple correspondence analysis

# Packages
library(FactoMineR)

# Depository
setwd(dir='C:/Users/lebre/OneDrive/Bureau/Projet')

############################################################################################################################

chiens <- read.csv("data/dogs",header = TRUE,sep=" ")
chiens

chiens[1:7] <- lapply(chiens[1:7],as.factor) # conversion in factors
summary(chiens)

acm_resultat <- MCA(chiens,quali.sup=7)
acm_resultat$eig # eigenvalues / information returned

plot.MCA(acm_resultat,choix="ind") # representation : 1st and 2nd principal components
plot.MCA(acm_resultat,choix="ind",axes= c(1,3))  # representation : 1st and 3rd principal components

# quality of representation
acm_resultat$var$cos2
acm_resultat$ind$cos2

# contributions
acm_resultat$var$contrib
acm_resultat$ind$contrib


