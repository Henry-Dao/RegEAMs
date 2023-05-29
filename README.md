# RegEAMs

This manual provides detailed instructions for executing the code to perform Bayesian estimation for LBA and DDMs with regressors. To simplify the process, I have organized and placed all the necessary code in separate folders, each corresponding to a specific section in [my paper](https://arxiv.org/abs/2302.10389) (coauthored with David Gunawan, Robert Kohn, Minh-Ngoc Tran, Guy Hawkins, and Scott Brown). For instance, if you wish to replicate the results presented in **Section 5.1** of the paper, please execute the code provided in the designated folder named "**Section_5\_1**". Within each folder, there are two main files: "**MAIN_CODE_MCMC.m**" and "**MAIN_CODE_VB.m**". You only need to run these .m files to perform the exact MCMC algorithm and the VB approximation, respectively. After successfully perform these algorithms, the results will be automatically stored as .mat files namely "*MCMC_result.mat*" and "*VB_result.mat*". These files will be served as inputs for the "**PLOTS.m**", which will generate the plots. Detailed explanations for each example are given below.

If you encounter any errors or issues, please notify me via email at [**viethung.unsw\@gmail.com**](mailto:viethung.unsw@gmail.com).

## Section_4

This folder comprises all the necessary Matlab files needed to replicate the simulation study for DDMs, as demonstrated in **Section 4** of the paper. To replicate the precise MCMC results, execute the "MAIN_CODE_MCMC.m" file. For reproducing the VB results, run the "MAIN_CODE_VB.m" file.

There are two datasets available: "*Simulated_data_medium.mat*" and "*Simulated_data_Large.mat*". Prior to executing the main code files, please select the desired dataset for replication.

Once you have successfully executed the VB and MCMC processes, the results will be saved in "VB_result.mat" and "MCMC_result.mat" respectively. Utilize these .mat files as inputs for the .m file "PLOTS.m" in order to generate the required plots.

## Section_5\_1

This folder contains all the necessary Matlab files needed to replicate the real data results for DDMs, as demonstrated in **Section 5.1** of the paper. To replicate the precise MCMC results, execute the "MAIN_CODE_MCMC.m" file. For reproducing the VB results, run the "MAIN_CODE_VB.m" file.

Once you have successfully executed the VB and MCMC processes, the results will be saved in "VB_result.mat" and "MCMC_result.mat" respectively. Utilize these .mat files as inputs for the .m file "PLOTS.m" in order to generate the required plots.

## Section_5\_2

This folder contains all the necessary Matlab files needed to replicate the results for the *Mental Rotation with Neural Covariates* as shown in **Section 5.2** of the paper.

For LBA models, please run the main .m files in **RegLBA** subfolder. For DDMs, the Matlab code can be found in **RegDDM** subfolder.

### Prediction

In this experiment, we need to generate simulated data from the posterior predictive distribution. It is straightforward for LBA models; you simply need to execute the .m file named "**PREDICTION_LBA.m**".

For DDMs, you will need to generate predictions using an R package called rtdists. To do this, follow these steps: First, run "**PREDICTION_DDM_part1.m**" to export the posterior draws. Then, execute "**PREDICTION_DDM_part2.R**" to generate predictive data from the DDM. Finally, run "**PREDICTION_DDM_part3.m**" to convert the predicted data to Matlab data format and generate plots. **Note that in "PREDICTION_DDM_part2.R", you need to set the working directory to the same location as the source file.**

## Section_5\_3

This folder contains all the necessary Matlab files needed to replicate the results for the *Human Connectome Project (HCP data)* as shown in **Section 5.3** of the paper.

For LBA models, please run the **"MAIN_CODE_VBL.m"** files in **RegLBA** subfolder. For DDMs, the Matlab code **"MAIN_CODE_VBL.m"** can be found in **RegDDM** subfolder.

To generate the plots, please run **"PLOTS.m"** provided in each subfolder. Please note that you must finish running VBL (or any relevant process) before creating the plots.
