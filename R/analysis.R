# Statistical Analysis and Visualization for ZANE
# R module for drug discovery metrics, distribution analysis, and plotting

#' Calculate ADMET Property Statistics
#'
#' Computes statistical summaries of ADMET properties across a molecular dataset.
#'
#' @param admet_scores Numeric vector of ADMET scores (0-1)
#' @param lipinski_violations Integer vector of violation counts
#' @param sa_scores Numeric vector of synthetic accessibility scores (1-10)
#'
#' @return List containing statistical summaries
#'
#' @examples
#' \dontrun{
#'   scores <- c(0.72, 0.85, 0.64, 0.91)
#'   stats <- calculate_admet_statistics(scores, c(0,0,1,0), c(7.2, 8.1, 6.5, 8.8))
#' }
#'
#' @export
calculate_admet_statistics <- function(admet_scores, lipinski_violations, sa_scores) {

  list(
    admet = list(
      mean = mean(admet_scores, na.rm = TRUE),
      median = median(admet_scores, na.rm = TRUE),
      sd = sd(admet_scores, na.rm = TRUE),
      min = min(admet_scores, na.rm = TRUE),
      max = max(admet_scores, na.rm = TRUE),
      q1 = quantile(admet_scores, 0.25, na.rm = TRUE),
      q3 = quantile(admet_scores, 0.75, na.rm = TRUE)
    ),
    lipinski = list(
      compliant = sum(lipinski_violations == 0),
      violations_mean = mean(lipinski_violations, na.rm = TRUE),
      violation_distribution = table(lipinski_violations)
    ),
    sa = list(
      mean = mean(sa_scores, na.rm = TRUE),
      median = median(sa_scores, na.rm = TRUE),
      sd = sd(sa_scores, na.rm = TRUE),
      easy_to_synthesize = sum(sa_scores > 7),
      hard_to_synthesize = sum(sa_scores < 3)
    ),
    sample_size = length(admet_scores)
  )
}

#' Analyze Hit Rate and Binding Affinity
#'
#' Statistical analysis of screening results and binding affinity distributions.
#'
#' @param hit_rates Numeric vector of hit rates (0-1) per experiment
#' @param binding_affinities Numeric vector of binding affinities (kcal/mol)
#' @param qed_scores Numeric vector of QED scores (0-1)
#'
#' @return List with analysis results
#'
#' @export
analyze_screening_results <- function(hit_rates, binding_affinities, qed_scores) {

  # Correlation analysis
  cor_affinity_qed <- cor(binding_affinities, qed_scores, use = "complete.obs")

  # Categorize binding affinities
  strong_binders <- sum(binding_affinities < -7, na.rm = TRUE)
  weak_binders <- sum(binding_affinities > -5, na.rm = TRUE)

  list(
    hit_rate_stats = list(
      mean_hit_rate = mean(hit_rates, na.rm = TRUE),
      pass_rate = sum(hit_rates > 0.15, na.rm = TRUE) / length(hit_rates),
      top_10_pct = quantile(hit_rates, 0.90, na.rm = TRUE)
    ),
    binding_affinity_stats = list(
      strong_binders = strong_binders,
      weak_binders = weak_binders,
      mean_affinity = mean(binding_affinities, na.rm = TRUE),
      best_affinity = min(binding_affinities, na.rm = TRUE)
    ),
    correlations = list(
      affinity_vs_qed = cor_affinity_qed
    )
  )
}

#' Compare Molecular Property Distributions
#'
#' Statistical comparison between two sets of molecules (e.g., hits vs non-hits).
#'
#' @param group1 Data frame of properties for group 1
#' @param group2 Data frame of properties for group 2
#' @param group1_name Name of first group
#' @param group2_name Name of second group
#'
#' @return List with comparison results and p-values
#'
#' @export
compare_molecule_groups <- function(group1, group2, group1_name = "Group1", group2_name = "Group2") {

  numeric_cols <- sapply(group1, is.numeric)
  comparisons <- list()

  for (col_name in names(numeric_cols)[numeric_cols]) {
    if (all(is.na(group1[[col_name]])) || all(is.na(group2[[col_name]]))) {
      next
    }

    # Perform t-test
    t_test <- t.test(group1[[col_name]], group2[[col_name]])

    comparisons[[col_name]] <- list(
      group1_mean = mean(group1[[col_name]], na.rm = TRUE),
      group2_mean = mean(group2[[col_name]], na.rm = TRUE),
      p_value = t_test$p.value,
      significant = t_test$p.value < 0.05
    )
  }

  comparisons
}

#' Predict Model Performance Trends
#'
#' Fit trends to model performance metrics over epochs.
#'
#' @param epochs Integer vector of epoch numbers
#' @param train_loss Numeric vector of training losses
#' @param val_loss Numeric vector of validation losses
#'
#' @return List with trend analysis
#'
#' @export
analyze_training_trends <- function(epochs, train_loss, val_loss) {

  # Fit linear models
  train_fit <- lm(train_loss ~ epochs)
  val_fit <- lm(val_loss ~ epochs)

  # Calculate convergence metrics
  last_n <- min(10, length(epochs))
  train_recent_std <- sd(train_loss[-(1:(length(train_loss)-last_n))], na.rm = TRUE)
  val_recent_std <- sd(val_loss[-(1:(length(val_loss)-last_n))], na.rm = TRUE)

  # Detect overfitting
  val_train_ratio <- mean(val_loss, na.rm = TRUE) / mean(train_loss, na.rm = TRUE)
  overfitting <- val_train_ratio > 1.2

  list(
    train_trend = list(
      coef = coef(train_fit)[2],
      r_squared = summary(train_fit)$r.squared,
      recent_stability = train_recent_std
    ),
    val_trend = list(
      coef = coef(val_fit)[2],
      r_squared = summary(val_fit)$r.squared,
      recent_stability = val_recent_std
    ),
    convergence = list(
      val_train_ratio = val_train_ratio,
      overfitting_detected = overfitting,
      recommendation = ifelse(overfitting, "Consider early stopping", "Training stable")
    )
  )
}

#' Recommender System for Drug Candidates
#'
#' Score and rank drug candidates based on multi-objective criteria.
#'
#' @param candidates Data frame with molecular properties and predictions
#' @param weights Named numeric vector with scoring weights
#'
#' @return Data frame with candidates ranked by combined score
#'
#' @export
rank_drug_candidates <- function(candidates, weights = NULL) {

  if (is.null(weights)) {
    weights <- c(
      admet_score = 0.25,
      qed = 0.20,
      sa_ease = 0.15,
      binding = 0.20,
      novelty = 0.10,
      solubility = 0.10
    )
  }

  # Normalize numeric columns to 0-1
  normalized <- candidates
  for (col in names(candidates)) {
    if (is.numeric(candidates[[col]])) {
      min_val <- min(candidates[[col]], na.rm = TRUE)
      max_val <- max(candidates[[col]], na.rm = TRUE)
      if (max_val > min_val) {
        normalized[[col]] <- (candidates[[col]] - min_val) / (max_val - min_val)
      }
    }
  }

  # Calculate combined score
  score <- rep(0, nrow(normalized))
  for (metric in names(weights)) {
    if (metric %in% names(normalized)) {
      score <- score + weights[metric] * normalized[[metric]]
    }
  }

  # Add score and rank
  result <- cbind(candidates, combined_score = score)
  result$rank <- rank(-result$combined_score)

  result[order(result$combined_score, decreasing = TRUE), ]
}
