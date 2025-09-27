library(dr4pl)
library(dplyr)

compute_log.ic50 <- function(l, u, ec50, h, md, MD) {
  if ((l >= 0.5) | (u <= 0.5)) {
    return(NA)
  } else {
    f1 <- function(x) {
      l + (u - l) / (1 + (2^x / ec50)^h) - 0.5
    }
    return(tryCatch(uniroot(f1, c(log2(md), log2(MD)))$root,
                    error = function(e) NA))
  }
}

compute_ic50 <- function(dose, viability) {
  # 新增：如果所有viability都<=0.5，直接返回最小剂量的log2
  if (all(viability <= 0.5)) {
    return(log2(min(dose)))
  }
  
  tryCatch({
    fit <- dr4pl(dose = dose,
                 response = viability,
                 method.init = "logistic",
                 trend = "decreasing")
    param <- as.numeric(fit$parameters)
    
    upper_limit <- param[1]
    ec50 <- param[2]
    slope <- -param[3]
    lower_limit <- param[4]
    
    MD <- max(dose)
    md <- min(dose)
    
    compute_log.ic50(lower_limit, upper_limit, ec50, slope, md, MD)
  }, error = function(e) {
    # 新增：如果拟合失败，但所有viability都<=0.5，返回log2(min(dose))
    # 注意：由于前面已经检查过all(viability <=0.5)的情况，这里无需再次检查
    NA
  })
}

data <- read.csv("./secondary_dose_value.csv")

results <- data %>%
  group_by(drug, cell_line) %>%
  summarise(
    ic50 = tryCatch(compute_ic50(dose, viability),
                    error = function(e) NA),
    .groups = "drop"
  )

write.csv(results, file = "secondary_ic50.csv", row.names = F)
print("Done")