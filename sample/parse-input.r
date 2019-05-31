library(rjson)

lines <- readLines("dataset-sample.json")

df <- as.data.frame(t(sapply(Lines, fromJSON)))