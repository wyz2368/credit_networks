# Define the list of variables
sim_per_profile_list = [500]
default_cost_list = [0.1, 0.3, 0.5, 0.7, 0.9]
utility_type_list = ["Bank_asset"]
is_eval_list = [False]
num_rounds_list = [10]
instance_path_list = ["./instances/merge/networks_merge10banks_1000ins_2070ext.pkl"]
# instance_path_list = ["./instances/merge/networks_merge10banks_1000ins_4070ext.pkl"]
# instance_path_list = ["./instances/merge/networks_merge10banks_1000ins_40100ext.pkl"]
root_result_folder_list = ["experiments_initial"]
merge_cost_factor_list = [0.05]
control_bonus_factor_list = [0.07]
file_name = "../runbatch01.sh"

# Generate the bash script content
bash_script_content = ""

for sim_per_profile in sim_per_profile_list:
    for default_cost in default_cost_list:
        for utility_type in utility_type_list:
            for is_eval in is_eval_list:
                for num_rounds in num_rounds_list:
                    for instance_path in instance_path_list:
                        for root_result_folder in root_result_folder_list:
                            for merge_cost_factor in merge_cost_factor_list:
                                for control_bonus_factor in control_bonus_factor_list:
                                    bash_script_content += (
                                        f"python example_merge.py --sim_per_profile {sim_per_profile} "
                                        f"--default_cost {default_cost} "
                                        f"--utility_type {utility_type} "
                                        f"--is_eval {is_eval} "
                                        f"--num_rounds {num_rounds} "
                                        f"--instance_path {instance_path} "
                                        f"--root_result_folder {root_result_folder} "
                                        f"--merge_cost_factor {merge_cost_factor} "
                                        f"--control_bonus_factor {control_bonus_factor}  && \\\n"
                                    )

# Remove the last '&& \\' from the script content
bash_script_content = bash_script_content.rstrip(' && \\\n')

# Save the bash script to a file
with open(file_name, "w") as file:
    file.write(bash_script_content)
