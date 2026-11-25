# ============================================================
#  K10 Psychiatric Disorder Prediction — Shiny App
#  Predicts lifetime and past-30-day disorder probabilities
#  from the K10 Psychological Distress Scale score.
# ============================================================

# ---- Packages -----------------------------------------------
required_pkgs <- c("shiny", "ggplot2", "scales", "DT")
to_install <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")

library(shiny)
library(ggplot2)
library(scales)
library(DT)

# ---- Constants ----------------------------------------------

DISORDER_LABELS <- c(
  MDE = "Major Depressive Episode",
  GAD = "Generalized Anxiety Disorder",
  PDS = "Panic Disorder",
  PTS = "Post-Traumatic Stress Disorder",
  OCD = "Obsessive-Compulsive Disorder",
  DYS = "Dysthymia (Persistent Depressive Disorder)",
  BP  = "Bipolar Disorder",
  SO  = "Social Phobia (Social Anxiety Disorder)"
)

DISORDER_COLORS <- c(
  MDE = "#1565C0", GAD = "#2E7D32", PDS = "#E65100",
  PTS = "#B71C1C", OCD = "#4527A0", DYS = "#00695C",
  BP  = "#880E4F", SO  = "#F57F17"
)

BAR_COLOR <- "#1A237E"

# Helper: extract short disorder key from outcome name (e.g. "D_MDE30" -> "MDE")
dx_short <- function(x) gsub("D_|30|DSM_", "", x)

# ---- Load Models at Startup ---------------------------------

# Shiny sets working directory to app folder; use relative paths
MODEL_DIR_30 <- "k10_model_objects_30_day"
MODEL_DIR_LT <- "k10_model_objects_lifetime"

bp30  <- readRDS(file.path(MODEL_DIR_30, "blueprint.rds"))
bplt  <- readRDS(file.path(MODEL_DIR_LT, "blueprint.rds"))

models_30 <- setNames(
  lapply(bp30$outcomes,  function(dx) readRDS(file.path(MODEL_DIR_30, sprintf("model_%s.rds", dx)))),
  bp30$outcomes
)
models_lt <- setNames(
  lapply(bplt$outcomes, function(dx) readRDS(file.path(MODEL_DIR_LT, sprintf("model_%s.rds", dx)))),
  bplt$outcomes
)

# TODO: set to 90th percentile of derivation sample K10 distribution
k10_high_threshold <- NA

# ---- Prediction Helpers -------------------------------------

predict_lr <- function(model_obj, X_new) {
  X_int <- cbind(intercept = 1, X_new)
  mc <- model_obj$feature_cols
  miss <- setdiff(mc, colnames(X_int))
  if (length(miss)) {
    pad   <- matrix(0, nrow(X_int), length(miss), dimnames = list(NULL, miss))
    X_int <- cbind(X_int, pad)
  }
  X_int <- X_int[, mc, drop = FALSE]
  lp    <- as.vector(X_int %*% model_obj$beta)
  1 / (1 + exp(-lp))
}

predict_en <- function(model_obj, X_new) {
  as.vector(predict(model_obj$fit, newx = X_new, s = model_obj$lambda, type = "response"))
}

predict_xgb <- function(model_obj, X_new) {
  predict(model_obj$fit, xgboost::xgb.DMatrix(X_new))
}

dispatch_predict <- function(model_obj, X_new) {
  switch(model_obj$type,
    lr  = predict_lr(model_obj, X_new),
    en  = predict_en(model_obj, X_new),
    xgb = predict_xgb(model_obj, X_new)
  )
}

prepare_new_data <- function(new_df, blueprint) {
  raw_preds   <- blueprint$predictors
  missing_c   <- setdiff(raw_preds, names(new_df))
  if (length(missing_c)) stop(paste("Missing columns:", paste(missing_c, collapse = ", ")))

  for (col in names(blueprint$factor_levels)) {
    if (col %in% names(new_df)) {
      lvls     <- blueprint$factor_levels[[col]]
      new_vals <- as.character(new_df[[col]])
      unseen   <- setdiff(unique(new_vals[!is.na(new_vals)]), lvls)
      if (length(unseen)) new_vals[new_vals %in% unseen] <- NA
      new_df[[col]] <- factor(new_vals, levels = lvls)
    }
  }

  fmla  <- as.formula(paste("~ 0 +", paste(raw_preds, collapse = " + ")))
  X_new <- suppressWarnings(model.matrix(fmla, data = new_df))

  tc   <- blueprint$feature_cols
  miss <- setdiff(tc, colnames(X_new))
  if (length(miss)) {
    pad   <- matrix(0, nrow(X_new), length(miss), dimnames = list(NULL, miss))
    X_new <- cbind(X_new, pad)
  }
  X_new[, tc, drop = FALSE]
}

run_predictions <- function(new_df, models, blueprint) {
  X <- prepare_new_data(new_df, blueprint)
  result <- sapply(blueprint$outcomes, function(dx) {
    p <- dispatch_predict(models[[dx]], X)
    pmax(pmin(p, 1 - 1e-7), 1e-7)
  })
  if (is.vector(result)) result <- matrix(result, nrow = 1, dimnames = list(NULL, blueprint$outcomes))
  as.data.frame(result)
}

bootstrap_mean_ci <- function(probs, B = 1000, alpha = 0.05) {
  n    <- length(probs)
  boot <- replicate(B, mean(sample(probs, n, replace = TRUE)))
  c(lower = unname(quantile(boot, alpha / 2)),
    upper = unname(quantile(boot, 1 - alpha / 2)))
}

# Build tidy data frame for plotting from a named vector of probabilities
probs_to_plot_df <- function(probs_named, prevalences) {
  short  <- dx_short(names(probs_named))
  labels <- unname(DISORDER_LABELS[short])
  data.frame(
    outcome = factor(labels, levels = rev(labels)),
    short   = short,
    prob    = as.numeric(probs_named),
    prev    = as.numeric(prevalences[names(probs_named)]),
    stringsAsFactors = FALSE
  )
}

plot_probs <- function(df, title) {
  ggplot(df, aes(x = prob, y = outcome)) +
    geom_col(width = 0.65, fill = BAR_COLOR, colour = "white") +
    geom_point(aes(x = prev), shape = 21, size = 3,
               fill = "white", colour = "#333333", stroke = 1.2) +
    geom_text(aes(label = paste0(round(prob * 100, 1), "%")),
              hjust = -0.15, size = 3.5, colour = "#222222") +
    scale_x_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, max(df$prob) * 1.25 + 0.05),
                       expand = c(0, 0)) +
    labs(title = title,
         subtitle = "Circle markers show training sample prevalence",
         x = "Predicted probability", y = NULL) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      legend.position    = "none",
      plot.title         = element_text(face = "bold"),
      axis.text.y        = element_text(colour = "#222222")
    )
}

# Build display table from wide predictions data frame
make_display_table <- function(preds_wide, blueprint) {
  out_cols <- blueprint$outcomes
  short    <- dx_short(out_cols)
  labels   <- DISORDER_LABELS[short]
  prevs    <- as.numeric(blueprint$prevalences[out_cols])

  rows <- lapply(seq_along(out_cols), function(i) {
    dx    <- out_cols[i]
    p     <- preds_wide[[dx]]
    prev  <- prevs[i]
    ratio <- p / prev
    label <- unname(labels[i])

    interp <- if (ratio > 1.1) {
      sprintf(
        "This individual's predicted probability of %s (%.1f%%) is approximately %.1f× the estimated population prevalence (%.1f%%).",
        label, p * 100, ratio, prev * 100)
    } else if (ratio >= 0.9) {
      sprintf(
        "This individual's predicted probability of %s (%.1f%%) is broadly consistent with the population average (%.1f%%).",
        label, p * 100, prev * 100)
    } else {
      sprintf(
        "This individual's predicted probability of %s (%.1f%%) is below the population average (%.1f%%).",
        label, p * 100, prev * 100)
    }

    data.frame(
      Disorder                    = label,
      `Predicted Prob (%)`        = paste0(formatC(p    * 100, format = "f", digits = 1), "%"),
      `Population Prev (%)`       = paste0(formatC(prev * 100, format = "f", digits = 1), "%"),
      `Relative Ratio`            = paste0(formatC(ratio, format = "f", digits = 1), "× population average"),
      `Interpretation`            = interp,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

# ---- Interpretation guide panel (shared) --------------------

interp_guide_panel <- tags$details(
  style = "margin-top:18px;",
  tags$summary(
    style = "cursor:pointer; font-weight:bold; color:#1A237E; padding:10px 14px;
             background:#E8EAF6; border-radius:6px; list-style:none; display:flex;
             align-items:center; gap:8px;",
    icon("circle-question"), " How to interpret these probabilities"
  ),
  div(style = "padding:16px 18px; border:1px solid #C5CAE9; border-top:none;
               border-radius:0 0 6px 6px; background:#FAFAFE;",
    h5(style = "margin-top:0; color:#1A237E;", "What the predicted probability means"),
    p("The predicted probability represents the expected proportion of people with this K10 score ",
      "in the general population who would meet diagnostic criteria for that condition. ",
      "It is not a binary diagnosis — it is a continuous estimate of risk given observed distress level."),
    tags$hr(),
    h5(style = "color:#1A237E;", "How to read the ratio to population prevalence"),
    tags$ul(
      tags$li("A ratio of ", strong("1.0×"), " means the individual's risk matches the population average."),
      tags$li("A ratio of ", strong("2.0×"), " means double the population-average risk for someone with this K10 score."),
      tags$li("A ratio of ", strong("0.5×"), " means half the population-average risk.")
    ),
    p("The population prevalence (shown as a circle marker on the chart) provides context for whether ",
      "the predicted probability is elevated, typical, or below average."),
    tags$hr(),
    h5(style = "color:#1A237E;", "Key assumptions and limitations"),
    tags$ul(
      tags$li("The model was calibrated on a general population sample and may not transfer perfectly to clinical or help-seeking populations."),
      tags$li("Predicted probabilities at very high or very low K10 scores carry greater uncertainty due to sparse data in the training sample at the extremes."),
      tags$li("If the individual is currently receiving treatment, their K10 score may understate their underlying disorder status."),
      tags$li("In clinical samples, elevated K10 scores may reflect physical illness or situational distress rather than the disorders the model was trained to detect."),
      tags$li("These are screening-level probability estimates — they are not diagnostic conclusions.")
    ),
    div(class = "alert alert-info", style = "margin-top:12px; margin-bottom:0;",
      icon("user-doctor"), strong(" Clinical interpretation:"),
      " All results should be interpreted by a qualified clinician in the context of a full clinical assessment."
    )
  )
)

# ---- Shared UI pieces ---------------------------------------

instructions_30 <- tagList(
  div(class = "alert alert-info",
    h4(icon("circle-info"), " How to use — Past 30-Day Predictions"),
    tags$ul(
      tags$li("Enter your K10 total score (sum of all 10 items, range 0–40)."),
      tags$li("Click ", strong("Generate Predictions"), " to see predicted probabilities for 8 psychiatric disorders in the past 30 days."),
      tags$li("Bars show your predicted probability for each disorder.")
    )
  )
)

instructions_lt <- tagList(
  div(class = "alert alert-info",
    h4(icon("circle-info"), " How to use — Lifetime Predictions"),
    tags$ul(
      tags$li("Enter your K10 total score and the demographic covariates below."),
      tags$li("Click ", strong("Generate Predictions"), " to see lifetime predicted probabilities for 8 psychiatric disorders."),
      tags$li("Bars show your predicted probability for each disorder."),
      tags$li(strong("Sex coding:"), " '1 (Male)' and '2 (Female)' correspond to the coding used in the training dataset. Please verify this matches your data."),
      tags$li(strong("Note:"), " predictions are statistical estimates, not clinical diagnoses.")
    )
  )
)

upload_instructions_30 <- div(class = "alert alert-info",
  h4(icon("circle-info"), " File Upload Instructions — Past 30 Days"),
  tags$ul(
    tags$li("Upload a CSV file with one row per person."),
    tags$li("Required column: ", code("k10_sum"), " — K10 total score (numeric, 0–40)."),
    tags$li("Optional: an ID column (e.g. ", code("id"), " or ", code("participant_id"), ") — any non-required column is treated as an ID and preserved in the output."),
    tags$li("Download a template below if needed.")
  ),
  downloadButton("dl_template_30", "Download CSV Template", class = "btn-sm btn-outline-secondary")
)

upload_instructions_lt <- div(class = "alert alert-info",
  h4(icon("circle-info"), " File Upload Instructions — Lifetime"),
  tags$ul(
    tags$li("Upload a CSV file with one row per person."),
    tags$li("Required columns:"),
    tags$ul(
      tags$li(code("k10_sum"), " — K10 total score (numeric, 0–40)"),
      tags$li(code("sex"), " — 1 = Male, 2 = Female"),
      tags$li(code("age"), " — age in years (numeric)"),
      tags$li(code("education"), " — 0 = No post-school qualifications, 1 = Post-school qualifications"),
      tags$li(code("employment"), " — 0 = Not employed, 1 = Employed"),
      tags$li(code("marital_status"), " — 0 = Not married/partnered, 1 = Married/partnered")
    ),
    tags$li("Download a template below if needed.")
  ),
  downloadButton("dl_template_lt", "Download CSV Template", class = "btn-sm btn-outline-secondary")
)

# ---- Custom CSS ---------------------------------------------

custom_css <- tags$style(HTML("
  body { background-color: #f8f9fa; }
  .navbar { background-color: #1A237E !important; }
  .navbar-brand, .navbar-nav > li > a { color: #ffffff !important; }
  .navbar-nav > li.active > a { background-color: #283593 !important; }
  .tab-content { padding-top: 20px; }
  .well { border-radius: 8px; background: #fff; border: 1px solid #dee2e6; padding: 20px; }
  .alert-info { background-color: #E3F2FD; border-color: #90CAF9; color: #0D47A1; border-radius: 8px; }
  .predict-btn { background-color: #1A237E; color: white; border: none;
                 padding: 10px 24px; border-radius: 6px; font-size: 15px; margin-top: 10px; }
  .predict-btn:hover { background-color: #283593; color: white; }
  .result-box { background: #fff; border-radius: 8px; padding: 20px;
                border: 1px solid #dee2e6; margin-top: 20px; }
  h3.section-title { color: #1A237E; border-bottom: 2px solid #1A237E;
                     padding-bottom: 6px; margin-bottom: 20px; }
  .disclaimer { font-size: 12px; color: #666; margin-top: 10px; font-style: italic; }
  .nav-tabs > li.active > a { color: #1A237E !important; font-weight: bold; }
  details > summary::-webkit-details-marker { display: none; }
  details[open] > summary { border-radius: 6px 6px 0 0; }
  .alert-warning { background-color: #FFF8E1; border-color: #FFE082; color: #5D4037;
                   border-radius: 8px; padding: 10px 14px; margin-top: 10px; }
  .calib-warn { font-size: 13px; }
"))

# ---- UI -----------------------------------------------------

ui <- navbarPage(
  title       = "K10 Disorder Prediction",
  id          = "main_nav",
  collapsible = TRUE,
  header      = custom_css,
  footer      = div(style = "text-align:center; padding:12px; color:#888; font-size:12px;",
    "Statistical predictions only — not a clinical diagnosis tool. ",
    "Refer to the accompanying paper for model details and performance metrics."
  ),

  # ===========================================================
  # TAB 1: PAST 30 DAYS
  # ===========================================================
  tabPanel("Past 30-Day Diagnoses",
    fluidPage(
      h3(class = "section-title", "Predicted Probabilities — Past 30 Days"),
      tabsetPanel(id = "tabs_30",
        # --- Manual entry ---
        tabPanel("Manual Entry",
          br(),
          fluidRow(
            column(4,
              div(class = "well",
                h4("Enter K10 Score"),
                p("Move the slider to your K10 total score."),
                sliderInput("k10_manual_30", "K10 Total Score (0–40)",
                            min = 0, max = 40, value = 10, step = 1),
                tags$hr(),
                div(
                  style = "background:#f1f3f4; border-radius:6px; padding:10px; margin-bottom:10px;",
                  strong("K10 Score Guide:"),
                  tags$ul(style = "margin:5px 0 0 0; padding-left:18px;",
                    tags$li("0–15: Likely to be well"),
                    tags$li("16–21: Mild distress"),
                    tags$li("22–29: Moderate distress"),
                    tags$li("30–40: Severe distress")
                  )
                ),
                uiOutput("warn_k10_30"),
                actionButton("predict_30_manual", "Generate Predictions",
                             class = "predict-btn", icon = icon("chart-bar")),
                p(class = "disclaimer",
                  "Predictions based on logistic regression models trained on a population sample.")
              )
            ),
            column(8,
              instructions_30,
              conditionalPanel("input.predict_30_manual > 0",
                div(class = "result-box",
                  plotOutput("plot_30_manual", height = "360px"),
                  br(),
                  DTOutput("table_30_manual"),
                  interp_guide_panel
                )
              )
            )
          )
        ),
        # --- File upload ---
        tabPanel("File Upload",
          br(),
          fluidRow(
            column(4,
              div(class = "well",
                h4("Upload Data File"),
                upload_instructions_30,
                br(),
                fileInput("file_30", "Choose CSV File",
                          accept = c("text/csv", ".csv"),
                          placeholder = "No file selected"),
                checkboxInput("header_30", "File has header row", value = TRUE),
                selectInput("sep_30", "Column separator",
                            choices = c(Comma = ",", Semicolon = ";", Tab = "\t"),
                            selected = ","),
                actionButton("predict_30_file", "Generate Predictions",
                             class = "predict-btn", icon = icon("chart-bar")),
                br(), br(),
                conditionalPanel("input.predict_30_file > 0",
                  downloadButton("dl_results_30", "Download Results (CSV)",
                                 class = "btn btn-success")
                )
              )
            ),
            column(8,
              conditionalPanel("output.file_30_loaded",
                h4("Data Preview"),
                div(class = "result-box",
                  DTOutput("preview_30"),
                  br(),
                  conditionalPanel("input.predict_30_file > 0",
                    h4("Prediction Results"),
                    DTOutput("results_30_file")
                  )
                )
              )
            )
          )
        )
      )
    )
  ),

  # ===========================================================
  # TAB 2: LIFETIME
  # ===========================================================
  tabPanel("Lifetime Diagnoses",
    fluidPage(
      h3(class = "section-title", "Predicted Probabilities — Lifetime Diagnoses"),
      tabsetPanel(id = "tabs_lt",
        # --- Manual entry ---
        tabPanel("Manual Entry",
          br(),
          fluidRow(
            column(4,
              div(class = "well",
                h4("Enter K10 Score & Demographics"),
                sliderInput("k10_manual_lt", "K10 Total Score (0–40)",
                            min = 0, max = 40, value = 10, step = 1),
                tags$hr(),
                selectInput("sex_lt", "Sex",
                            choices  = c("Male" = "1", "Female" = "2"),
                            selected = "1"),
                numericInput("age_lt", "Age (years)", value = 40, min = 15, max = 100, step = 1),
                selectInput("education_lt", "Highest education",
                            choices  = c("No post-school qualifications" = "No post-school",
                                         "Post-school qualifications"     = "Post-school"),
                            selected = "No post-school"),
                selectInput("employment_lt", "Current employment status",
                            choices  = c("Not employed" = "0", "Employed" = "1"),
                            selected = "0"),
                selectInput("marital_lt", "Marital / relationship status",
                            choices  = c("Not married / not partnered" = "0",
                                         "Married / partnered"         = "1"),
                            selected = "0"),
                tags$hr(),
                uiOutput("warn_k10_lt"),
                actionButton("predict_lt_manual", "Generate Predictions",
                             class = "predict-btn", icon = icon("chart-bar")),
                p(class = "disclaimer",
                  "Sex coding (1=Male, 2=Female) matches training data. ",
                  "Verify this matches your dataset.")
              )
            ),
            column(8,
              instructions_lt,
              conditionalPanel("input.predict_lt_manual > 0",
                div(class = "result-box",
                  plotOutput("plot_lt_manual", height = "360px"),
                  br(),
                  DTOutput("table_lt_manual"),
                  interp_guide_panel
                )
              )
            )
          )
        ),
        # --- File upload ---
        tabPanel("File Upload",
          br(),
          fluidRow(
            column(4,
              div(class = "well",
                h4("Upload Data File"),
                upload_instructions_lt,
                br(),
                fileInput("file_lt", "Choose CSV File",
                          accept = c("text/csv", ".csv"),
                          placeholder = "No file selected"),
                checkboxInput("header_lt", "File has header row", value = TRUE),
                selectInput("sep_lt", "Column separator",
                            choices = c(Comma = ",", Semicolon = ";", Tab = "\t"),
                            selected = ","),
                actionButton("predict_lt_file", "Generate Predictions",
                             class = "predict-btn", icon = icon("chart-bar")),
                br(), br(),
                conditionalPanel("input.predict_lt_file > 0",
                  downloadButton("dl_results_lt", "Download Results (CSV)",
                                 class = "btn btn-success")
                )
              )
            ),
            column(8,
              conditionalPanel("output.file_lt_loaded",
                h4("Data Preview"),
                div(class = "result-box",
                  DTOutput("preview_lt"),
                  br(),
                  conditionalPanel("input.predict_lt_file > 0",
                    h4("Prediction Results"),
                    DTOutput("results_lt_file")
                  )
                )
              )
            )
          )
        )
      )
    )
  ),

  # ===========================================================
  # TAB 3: GROUP PREVALENCE ESTIMATION
  # ===========================================================
  tabPanel("Group Prevalence Estimation",
    fluidPage(
      h3(class = "section-title", "Group Prevalence Estimation — Past 30-Day Disorders"),
      fluidRow(
        column(4,
          div(class = "well",
            h4("Upload Group K10 Data"),
            div(class = "alert alert-info",
              h4(icon("circle-info"), " Instructions"),
              tags$ul(
                tags$li("Upload a CSV file containing K10 scores for your group (one row per person)."),
                tags$li("Specify the column name containing the K10 total scores (default: ", code("k10_sum"), ")."),
                tags$li("The app will apply the past-30-day models to all rows and compute group-level prevalence estimates with 95% bootstrap confidence intervals (1,000 resamples)."),
                tags$li("Note: only K10 score is required — this uses the past-30-day models.")
              )
            ),
            fileInput("file_grp", "Choose CSV File",
                      accept = c("text/csv", ".csv"),
                      placeholder = "No file selected"),
            textInput("grp_col_name", "K10 score column name", value = "k10_sum"),
            actionButton("predict_grp", "Estimate Group Prevalence",
                         class = "predict-btn", icon = icon("chart-bar")),
            br(), br(),
            conditionalPanel("input.predict_grp > 0",
              downloadButton("dl_grp_results", "Download Summary (CSV)",
                             class = "btn btn-success")
            )
          )
        ),
        column(8,
          conditionalPanel("output.grp_file_loaded",
            div(class = "alert alert-info",
              style = "font-size:13px;",
              icon("triangle-exclamation"), " ",
              strong("Important:"),
              " Estimated prevalences are derived by applying a population-calibrated model to the uploaded K10 scores. ",
              "Estimates assume the K10–diagnosis relationship is transportable from the general population derivation sample. ",
              "Treat as screening-level estimates only."
            ),
            div(class = "result-box",
              uiOutput("grp_n_display"),
              br(),
              conditionalPanel("input.predict_grp > 0",
                h4("Group Prevalence Summary"),
                DTOutput("grp_results_table")
              )
            )
          )
        )
      )
    )
  ),

  # ===========================================================
  # TAB 4: HELP / ABOUT
  # ===========================================================
  tabPanel("Help & About",
    fluidPage(
      fluidRow(
        column(8, offset = 2,
          br(),
          h3(class = "section-title", "About This Tool"),
          div(class = "well",
            h4("What does this tool do?"),
            p("This app uses logistic regression models trained on a population sample to predict
               the probability that a person meets diagnostic criteria for each of 8 psychiatric disorders,
               based on their K10 Psychological Distress Scale score and (for lifetime diagnoses)
               demographic characteristics."),
            tags$hr(),
            h4("The K10 Scale"),
            p("The Kessler Psychological Distress Scale (K10) is a 10-item questionnaire measuring
               non-specific psychological distress. Scores range from 0 (no distress) to 40 (severe distress).
               Higher scores indicate greater distress."),
            div(class = "alert alert-info",
              strong("Scoring guide:"),
              tags$ul(
                tags$li("0–15: Likely to be well"),
                tags$li("16–21: Mild psychological distress"),
                tags$li("22–29: Moderate psychological distress"),
                tags$li("30–40: Severe psychological distress")
              )
            ),
            tags$hr(),
            h4("Disorders Predicted"),
            DTOutput("disorder_table"),
            tags$hr(),
            h4("Model Details"),
            tags$ul(
              tags$li(strong("Past 30-Day model:"), " Logistic regression using K10 total score only."),
              tags$li(strong("Lifetime model:"), " Logistic regression using K10 total score, sex, age, education, employment status, and marital status.")
            ),
            tags$hr(),
            h4("CSV File Format"),
            h5("Past 30-Day Upload"),
            p("Minimum required column:"),
            tags$table(class = "table table-bordered table-sm",
              tags$thead(tags$tr(tags$th("Column"), tags$th("Type"), tags$th("Values"))),
              tags$tbody(
                tags$tr(tags$td(code("k10_sum")), tags$td("Numeric"), tags$td("0 to 40"))
              )
            ),
            h5("Lifetime Upload"),
            p("Required columns:"),
            tags$table(class = "table table-bordered table-sm",
              tags$thead(tags$tr(tags$th("Column"), tags$th("Type"), tags$th("Values"))),
              tags$tbody(
                tags$tr(tags$td(code("k10_sum")),       tags$td("Numeric"),    tags$td("0 to 40")),
                tags$tr(tags$td(code("sex")),           tags$td("Integer"),    tags$td("1 = Male, 2 = Female")),
                tags$tr(tags$td(code("age")),           tags$td("Numeric"),    tags$td("Age in years")),
                tags$tr(tags$td(code("education")),     tags$td("Integer"),    tags$td("0 = No post-school, 1 = Post-school")),
                tags$tr(tags$td(code("employment")),    tags$td("Integer"),    tags$td("0 = Not employed, 1 = Employed")),
                tags$tr(tags$td(code("marital_status")),tags$td("Integer"),    tags$td("0 = Not married/partnered, 1 = Married/partnered"))
              )
            ),
            tags$hr(),
            div(class = "alert alert-info",
              icon("triangle-exclamation"), strong(" Disclaimer:"),
              " Predictions are statistical estimates based on population-level models and should not be
              used as a substitute for clinical assessment or diagnosis. Consult the original publication
              for model performance metrics and appropriate use cases."
            )
          )
        )
      )
    )
  )
)

# ---- Server -------------------------------------------------

server <- function(input, output, session) {

  # ============================================================
  # CALIBRATION WARNINGS
  # ============================================================

  calib_warn_ui <- function(k10_val) {
    if (!is.na(k10_high_threshold) && !is.na(k10_val) && k10_val > k10_high_threshold) {
      div(class = "alert alert-warning calib-warn",
        icon("triangle-exclamation"), " ",
        "Note: This K10 score is in the upper range of the derivation sample. ",
        "Predicted probabilities at extreme scores carry greater uncertainty and should be interpreted cautiously."
      )
    }
  }

  output$warn_k10_30 <- renderUI(calib_warn_ui(input$k10_manual_30))
  output$warn_k10_lt <- renderUI(calib_warn_ui(input$k10_manual_lt))

  # ============================================================
  # CSV TEMPLATES
  # ============================================================

  output$dl_template_30 <- downloadHandler(
    filename = "k10_30day_template.csv",
    content  = function(f) {
      write.csv(data.frame(id = 1:3, k10_sum = c(12, 24, 36)), f, row.names = FALSE)
    }
  )

  output$dl_template_lt <- downloadHandler(
    filename = "k10_lifetime_template.csv",
    content  = function(f) {
      write.csv(data.frame(
        id             = 1:3,
        k10_sum        = c(12, 24, 36),
        sex            = c(1, 2, 1),
        age            = c(35, 50, 28),
        education      = c(1, 0, 1),
        employment     = c(1, 0, 1),
        marital_status = c(1, 0, 0)
      ), f, row.names = FALSE)
    }
  )

  # ============================================================
  # 30-DAY MANUAL
  # ============================================================

  results_30_manual <- eventReactive(input$predict_30_manual, {
    new_df <- data.frame(k10_sum = as.numeric(input$k10_manual_30))
    tryCatch(
      run_predictions(new_df, models_30, bp30),
      error = function(e) { showNotification(paste("Error:", e$message), type = "error"); NULL }
    )
  })

  output$plot_30_manual <- renderPlot({
    req(results_30_manual())
    preds <- results_30_manual()
    probs <- unlist(preds[1, ])
    df    <- probs_to_plot_df(probs, bp30$prevalences)
    plot_probs(df, paste0("Past 30-Day Predicted Probabilities  (K10 = ", input$k10_manual_30, ")"))
  })

  output$table_30_manual <- renderDT({
    req(results_30_manual())
    tbl <- make_display_table(results_30_manual(), bp30)
    datatable(tbl, rownames = FALSE, options = list(dom = "t", pageLength = 10),
              class = "table-striped table-hover")
  })

  # ============================================================
  # 30-DAY FILE UPLOAD
  # ============================================================

  data_30_raw <- reactive({
    req(input$file_30)
    tryCatch(
      read.csv(input$file_30$datapath, header = input$header_30,
               sep = input$sep_30, stringsAsFactors = FALSE),
      error = function(e) { showNotification(paste("Could not read file:", e$message), type = "error"); NULL }
    )
  })

  output$file_30_loaded <- reactive(!is.null(data_30_raw()))
  outputOptions(output, "file_30_loaded", suspendWhenHidden = FALSE)

  output$preview_30 <- renderDT({
    req(data_30_raw())
    datatable(head(data_30_raw(), 10), rownames = FALSE,
              options = list(dom = "t", scrollX = TRUE), class = "table-striped")
  })

  results_30_file <- eventReactive(input$predict_30_file, {
    req(data_30_raw())
    df <- data_30_raw()
    if (!"k10_sum" %in% names(df)) {
      showNotification("Column 'k10_sum' not found in uploaded file.", type = "error")
      return(NULL)
    }
    id_cols    <- setdiff(names(df), bp30$predictors)
    pred_input <- df[, bp30$predictors, drop = FALSE]
    preds      <- tryCatch(
      run_predictions(pred_input, models_30, bp30),
      error = function(e) { showNotification(paste("Prediction error:", e$message), type = "error"); NULL }
    )
    req(preds)
    # Rename outcome columns to readable labels
    short  <- dx_short(bp30$outcomes)
    labels <- paste0("prob_", short, "_30d")
    names(preds) <- labels
    cbind(if (length(id_cols)) df[, id_cols, drop = FALSE] else data.frame(row = seq_len(nrow(df))),
          df[, bp30$predictors, drop = FALSE],
          preds)
  })

  output$results_30_file <- renderDT({
    req(results_30_file())
    datatable(results_30_file(), rownames = FALSE,
              options = list(scrollX = TRUE, pageLength = 10), class = "table-striped table-hover") |>
      formatPercentage(grep("^prob_", names(results_30_file()), value = TRUE), digits = 1)
  })

  output$dl_results_30 <- downloadHandler(
    filename = function() paste0("k10_30day_predictions_", Sys.Date(), ".csv"),
    content  = function(f) {
      req(results_30_file())
      write.csv(results_30_file(), f, row.names = FALSE)
    }
  )

  # ============================================================
  # LIFETIME MANUAL
  # ============================================================

  results_lt_manual <- eventReactive(input$predict_lt_manual, {
    new_df <- data.frame(
      k10_sum        = as.numeric(input$k10_manual_lt),
      sex            = as.character(input$sex_lt),
      age            = as.numeric(input$age_lt),
      education      = as.character(input$education_lt),
      employment     = as.character(input$employment_lt),
      marital_status = as.character(input$marital_lt),
      stringsAsFactors = FALSE
    )
    tryCatch(
      run_predictions(new_df, models_lt, bplt),
      error = function(e) { showNotification(paste("Error:", e$message), type = "error"); NULL }
    )
  })

  output$plot_lt_manual <- renderPlot({
    req(results_lt_manual())
    preds <- results_lt_manual()
    probs <- unlist(preds[1, ])
    df    <- probs_to_plot_df(probs, bplt$prevalences)
    plot_probs(df, paste0("Lifetime Predicted Probabilities  (K10 = ", input$k10_manual_lt, ")"))
  })

  output$table_lt_manual <- renderDT({
    req(results_lt_manual())
    tbl <- make_display_table(results_lt_manual(), bplt)
    datatable(tbl, rownames = FALSE, options = list(dom = "t", pageLength = 10),
              class = "table-striped table-hover")
  })

  # ============================================================
  # LIFETIME FILE UPLOAD
  # ============================================================

  data_lt_raw <- reactive({
    req(input$file_lt)
    tryCatch(
      read.csv(input$file_lt$datapath, header = input$header_lt,
               sep = input$sep_lt, stringsAsFactors = FALSE),
      error = function(e) { showNotification(paste("Could not read file:", e$message), type = "error"); NULL }
    )
  })

  output$file_lt_loaded <- reactive(!is.null(data_lt_raw()))
  outputOptions(output, "file_lt_loaded", suspendWhenHidden = FALSE)

  output$preview_lt <- renderDT({
    req(data_lt_raw())
    datatable(head(data_lt_raw(), 10), rownames = FALSE,
              options = list(dom = "t", scrollX = TRUE), class = "table-striped")
  })

  results_lt_file <- eventReactive(input$predict_lt_file, {
    req(data_lt_raw())
    df          <- data_lt_raw()
    req_cols    <- bplt$predictors
    missing_c   <- setdiff(req_cols, names(df))
    if (length(missing_c)) {
      showNotification(paste("Missing columns:", paste(missing_c, collapse = ", ")), type = "error")
      return(NULL)
    }
    id_cols    <- setdiff(names(df), req_cols)
    pred_input <- df[, req_cols, drop = FALSE]
    # Convert education 0/1 integer to text labels expected by model
    if ("education" %in% names(pred_input)) {
      pred_input$education <- ifelse(as.integer(pred_input$education) == 1,
                                     "Post-school", "No post-school")
    }
    # Ensure character types for remaining factors
    for (col in c("sex", "employment", "marital_status")) {
      if (col %in% names(pred_input)) pred_input[[col]] <- as.character(pred_input[[col]])
    }
    preds <- tryCatch(
      run_predictions(pred_input, models_lt, bplt),
      error = function(e) { showNotification(paste("Prediction error:", e$message), type = "error"); NULL }
    )
    req(preds)
    short  <- dx_short(bplt$outcomes)
    labels <- paste0("prob_", short, "_lifetime")
    names(preds) <- labels
    cbind(if (length(id_cols)) df[, id_cols, drop = FALSE] else data.frame(row = seq_len(nrow(df))),
          df[, req_cols, drop = FALSE],
          preds)
  })

  output$results_lt_file <- renderDT({
    req(results_lt_file())
    datatable(results_lt_file(), rownames = FALSE,
              options = list(scrollX = TRUE, pageLength = 10), class = "table-striped table-hover") |>
      formatPercentage(grep("^prob_", names(results_lt_file()), value = TRUE), digits = 1)
  })

  output$dl_results_lt <- downloadHandler(
    filename = function() paste0("k10_lifetime_predictions_", Sys.Date(), ".csv"),
    content  = function(f) {
      req(results_lt_file())
      write.csv(results_lt_file(), f, row.names = FALSE)
    }
  )

  # ============================================================
  # GROUP PREVALENCE TAB
  # ============================================================

  data_grp_raw <- reactive({
    req(input$file_grp)
    tryCatch(
      read.csv(input$file_grp$datapath, header = TRUE, stringsAsFactors = FALSE),
      error = function(e) { showNotification(paste("Could not read file:", e$message), type = "error"); NULL }
    )
  })

  output$grp_file_loaded <- reactive(!is.null(data_grp_raw()))
  outputOptions(output, "grp_file_loaded", suspendWhenHidden = FALSE)

  output$grp_n_display <- renderUI({
    req(data_grp_raw())
    df     <- data_grp_raw()
    col_nm <- trimws(input$grp_col_name)
    if (col_nm %in% names(df)) {
      n_valid <- sum(!is.na(df[[col_nm]]))
      div(style = "color:#1A237E; font-weight:bold;",
        icon("users"), sprintf(" %d rows loaded, %d with valid K10 scores.", nrow(df), n_valid))
    } else {
      div(class = "alert alert-warning",
        icon("triangle-exclamation"),
        sprintf(" Column '%s' not found in the uploaded file.", col_nm))
    }
  })

  grp_results <- eventReactive(input$predict_grp, {
    req(data_grp_raw())
    df     <- data_grp_raw()
    col_nm <- trimws(input$grp_col_name)
    if (!col_nm %in% names(df)) {
      showNotification(paste0("Column '", col_nm, "' not found in file."), type = "error")
      return(NULL)
    }
    pred_df <- data.frame(k10_sum = as.numeric(df[[col_nm]]))
    pred_df <- pred_df[!is.na(pred_df$k10_sum), , drop = FALSE]
    if (nrow(pred_df) == 0) {
      showNotification("No valid K10 scores found after removing NAs.", type = "error")
      return(NULL)
    }
    preds <- tryCatch(
      run_predictions(pred_df, models_30, bp30),
      error = function(e) { showNotification(paste("Prediction error:", e$message), type = "error"); NULL }
    )
    req(preds)
    set.seed(42)
    rows <- lapply(bp30$outcomes, function(dx) {
      short  <- dx_short(dx)
      label  <- unname(DISORDER_LABELS[short])
      p      <- preds[[dx]]
      prev   <- as.numeric(bp30$prevalences[[dx]])
      mean_p <- mean(p, na.rm = TRUE)
      ci     <- bootstrap_mean_ci(p[!is.na(p)])
      ratio  <- mean_p / prev
      data.frame(
        Diagnosis                    = label,
        `Group Prevalence (%)`       = paste0(formatC(mean_p    * 100, format = "f", digits = 1), "%"),
        `95% CI`                     = paste0(formatC(ci["lower"] * 100, format = "f", digits = 1),
                                              "% – ",
                                              formatC(ci["upper"] * 100, format = "f", digits = 1), "%"),
        `Population Prevalence (%)`  = paste0(formatC(prev * 100, format = "f", digits = 1), "%"),
        `Ratio (Group / Population)` = formatC(ratio, format = "f", digits = 2),
        # raw columns for download
        mean_p_raw = mean_p, ci_lo_raw = ci["lower"], ci_hi_raw = ci["upper"],
        prev_raw = prev, ratio_raw = ratio,
        check.names = FALSE, stringsAsFactors = FALSE
      )
    })
    do.call(rbind, rows)
  })

  output$grp_results_table <- renderDT({
    req(grp_results())
    display_cols <- c("Diagnosis", "Group Prevalence (%)", "95% CI",
                      "Population Prevalence (%)", "Ratio (Group / Population)")
    datatable(grp_results()[, display_cols],
              rownames = FALSE,
              options  = list(dom = "t", pageLength = 10, autoWidth = TRUE),
              class    = "table-striped table-hover table-bordered")
  })

  output$dl_grp_results <- downloadHandler(
    filename = function() paste0("k10_group_prevalence_", Sys.Date(), ".csv"),
    content  = function(f) {
      req(grp_results())
      out <- grp_results()
      raw_cols <- grep("_raw$", names(out), value = TRUE)
      export   <- out[, setdiff(names(out), raw_cols)]
      write.csv(export, f, row.names = FALSE)
    }
  )

  # ============================================================
  # HELP TAB — DISORDER TABLE
  # ============================================================

  output$disorder_table <- renderDT({
    short  <- names(DISORDER_LABELS)
    prev30 <- as.numeric(bp30$prevalences[paste0("D_", short, "30")]) * 100
    prevlt <- as.numeric(bplt$prevalences[paste0("DSM_", short)]) * 100
    df <- data.frame(
      `Short code`      = short,
      `Full name`       = unname(DISORDER_LABELS),
      `30-Day prev (%)`  = formatC(prev30, format = "f", digits = 1),
      `Lifetime prev (%)` = formatC(prevlt, format = "f", digits = 1),
      check.names = FALSE
    )
    datatable(df, rownames = FALSE, options = list(dom = "t"), class = "table-bordered table-sm")
  })
}

# ---- Run App ------------------------------------------------
shinyApp(ui = ui, server = server)
