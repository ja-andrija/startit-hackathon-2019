library(jsonlite)
library(data.table)

lines <- readLines("dataset-sample.json")
something <- t(sapply(lines, fromJSON))

v <- c()

for (x in something) 
	v <- append(v, list(class = x$device_class, mac = x$mac))

dt <- data.table(v)

head(dt, 3)

ans <- dt[, .(.N), by = "v"]

ans