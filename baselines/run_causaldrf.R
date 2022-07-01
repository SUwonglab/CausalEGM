library(causaldrf)
source("simulator.R")

# Generate simulation dataset
Linear_sim_data <- linear_effect_Sim(N = 10000, v_dim=15)
Nolinear_sim_data <- nolinear_effect_Sim(N = 10000, v_dim=15)
Imbens_sim_data <- Imbens_Sim(N=10000, v_dim=15)

run_iptw <- function(dataset, start, end, nb_intervals=200){
    grid_eval <- seq(start, end, length.out = nb_intervals)
    true_effect = grid_eval + 2/((1 + grid_eval)**3)
    iptw_estimate <- iptw_est(Y = Y,
                            treat = X,
                            treat_formula = X ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15,
                            data = dataset,
                            numerator_formula = X~1,
                            degree=2,
                            treat_mod = "Normal")
                            #treat_mod = "Gamma",
                            #link_function = "inverse")
    predict_effect <- iptw_estimate$param[1] + iptw_estimate$param[2] * grid_eval + iptw_estimate$param[3] * (grid_eval**2)
    return(data.frame(true_effect, predict_effect))
}

run_reg <- function(dataset, start, end, nb_intervals=200){
    grid_eval <- seq(start, end, length.out = nb_intervals)
    true_effect = grid_eval + 2/((1 + grid_eval)**3)
    reg_estimate <- reg_est(Y = Y,
                            treat = X,
                            covar_formula = ~ .-Y-X,
                            covar_lin_formula = ~ .-Y-X,
                            covar_sq_formula = ~ .-Y-X,
                            data = dataset,
                            degree=2,
                            method = "same"
                            )
    predict_effect <- reg_estimate$param[1] + reg_estimate$param[2] * grid_eval + reg_estimate$param[3] * (grid_eval**2)
    return(data.frame(true_effect, predict_effect))
}

run_bart <- function(dataset, start, end, nb_intervals=200){
    grid_eval <- seq(start, end, length.out = nb_intervals)
    true_effect = grid_eval + 2/((1 + grid_eval)**3)
    bart_estimate <- bart_est(Y = Y,
                            treat = X,
                            outcome_formula = Y ~ X+V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15,
                            data = dataset,
                            grid_val = grid_eval
                            )
    predict_effect <- bart_estimate$out_mod$yhat.train.mean
    return(data.frame(true_effect, predict_effect))
}

reg_linear_out = run_reg(dataset=Linear_sim_data, start=-5, end=5)
reg_nolinear_out = run_reg(dataset=Nolinear_sim_data, start=-10, end=10)
reg_Imbens_out = run_reg(dataset=Imbens_sim_data, start=0, end=3)
write.table(reg_linear_out, file = "reg_linear_out.txt", sep = "\t")
write.table(reg_nolinear_out, file = "reg_nolinear_out.txt", sep = "\t")
write.table(reg_Imbens_out, file = "reg_Imbens_out.txt", sep = "\t")


iptw_linear_out = run_iptw(dataset=Linear_sim_data, start=-5, end=5)
iptw_nolinear_out = run_iptw(dataset=Nolinear_sim_data, start=-10, end=10)
iptw_Imbens_out = run_iptw(dataset=Imbens_sim_data, start=0, end=3)
write.table(iptw_linear_out, file = "iptw_linear_out.txt", sep = "\t")
write.table(iptw_nolinear_out, file = "iptw_nolinear_out.txt", sep = "\t")
write.table(iptw_Imbens_out, file = "iptw_Imbens_out.txt", sep = "\t")

bart_linear_out = run_bart(dataset=Linear_sim_data, start=-5, end=5)
bart_nolinear_out = run_bart(dataset=Nolinear_sim_data, start=-10, end=10)
bart_Imbens_out = run_bart(dataset=Imbens_sim_data, start=0, end=3)
write.table(bart_linear_out, file = "bart_linear_out.txt", sep = "\t")
write.table(bart_nolinear_out, file = "bart_nolinear_out.txt", sep = "\t")
write.table(bart_Imbens_out, file = "bart_Imbens_out.txt", sep = "\t")


