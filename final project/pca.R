library(glmnet)
library(ggfortify)

setwd("/Users/lmarbach/Library/CloudStorage/GoogleDrive-lmarbach@stanford.edu/My Drive/Coursework/Year 2/Fall 2023/CS 229/project")

df <- read.csv("parcoursup.csv")
df <- df[, -1]

#PCA

pca_grades <- prcomp(na.omit(df[, 27:71]), scale.=F)

coeff_table_grades <- data.frame(variable = colnames(df[, 27:71]),
                                 coefficient = pca_grades$rotation[, 1])
coeff_table_grades <- coeff_table_grades[order(abs(coeff_table_grades$coefficient), 
                                               decreasing = TRUE), ]
print(coeff_table_grades, row.names = FALSE)

df$Ranking_Quintile <- cut(df$Classement, breaks = 5, labels = FALSE)

pca_plot <- autoplot(pca_grades, data = na.omit(df[, c(27:71, 72)]), colour = "Ranking_Quintile") +
  theme_minimal() +  # or use any other theme you prefer
  theme(
    panel.grid.major = element_line(color = "gray", linetype = "dashed"), 
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

# Display the PCA plot
print(pca_plot)

autoplot(pca_grades, data = na.omit(df[, c(27:71, 72)]), colour = "Ranking_quintile")

pca_grades_scores <- pca_grades$x[, 1]
summary(aov(pca_grades_scores ~ na.omit(df[, c(27:71, 72)])$Ranking_quintile))