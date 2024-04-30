import re

# Path to the log file
log_file_path = '/home/chenyu/Desktop/DDRL_project/test/DDRL_project/bet/exp_local/2024.04.29/203857_kitchen_eval/run_on_env.log'

# Read the content of the log file
with open(log_file_path, 'r') as file:
    log_content = file.readlines()

# Regular expression to find lines containing "completed_tasks"
task_regex = re.compile(r"'completed_tasks':\s\{([^}]+)\}")

# Function to extract and treat each unique set of completed tasks as a single entry
def parse_task_sets(line):
    match = task_regex.search(line)
    if match:
        # Keep the entire set of tasks as one string
        return match.group(1).replace("'", "").strip()
    return None

# Parse the entire log file using the corrected approach
completed_task_sets = []
for line in log_content:
    task_set = parse_task_sets(line)
    if task_set:
        completed_task_sets.append(task_set)

# Count the frequency of each set of tasks
task_set_frequency = {}
for task_set in completed_task_sets:
    if task_set in task_set_frequency:
        task_set_frequency[task_set] += 1
    else:
        task_set_frequency[task_set] = 1

# Output the frequencies of task sets
for task, freq in task_set_frequency.items():
    print(f"{task}: {freq}")
