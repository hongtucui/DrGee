library(dr4pl)
library(dplyr)

# Calculate log2(IC50) based on the 4-parameter logistic model parameters
compute_log.ic50 <- function(l, u, ec50, h, md, MD) {
  # Return NA if the curve doesn't cross the 0.5 viability threshold
  if ((l >= 0.5) | (u <= 0.5)) {
    return(NA)
  } else {
    # Define the function to find where viability equals 0.5
    f1 <- function(x) {
      l + (u - l) / (1 + (2^x / ec50)^h) - 0.5
    }
    # Solve for x using root finding within the dose range
    return(tryCatch(uniroot(f1, c(log2(md), log2(MD)))$root,
                    error = function(e) NA))
  }
}

# Main function to compute IC50 from dose-response data
compute_ic50 <- function(dose, viability) {
  # New: If all viability values are <= 0.5, return log2 of the minimum dose
  if (all(viability <= 0.5)) {
    return(log2(min(dose)))
  }
  
  tryCatch({
    # Fit 4-parameter logistic model to dose-response data
    fit <- dr4pl(dose = dose,
                 response = viability,
                 method.init = "logistic",
                 trend = "decreasing")
    # Extract model parameters
    param <- as.numeric(fit$parameters)
    
    upper_limit <- param[1]  # Upper asymptote
    ec50 <- param[2]         # EC50 parameter from model
    slope <- -param[3]       # Slope parameter (converted to positive)
    lower_limit <- param[4]  # Lower asymptote
    
    MD <- max(dose)  # Maximum dose in the data
    md <- min(dose)  # Minimum dose in the data
    
    # Calculate log2(IC50) using the helper function
    compute_log.ic50(lower_limit, upper_limit, ec50, slope, md, MD)
  }, error = function(e) {
    # Return NA if model fitting fails
    # Note: We already checked the all(viability <= 0.5) case earlier, so no need to check here
    NA
  })
}

# Read input data containing dose-response measurements
data <- read.csv("./secondary_dose_value.csv")

# Calculate IC50 for each drug-cell line combination
results <- data %>%
  group_by(drug, cell_line) %>%
  summarise(
    ic50 = tryCatch(compute_ic50(dose, viability),
                    error = function(e) NA),
    .groups = "drop"
  )

# Save results to CSV file
write.csv(results, file = "secondary_ic50.csv", row.names = F)
print("Done")