import os
import glob

# Script to recover the results from the log files
def get_accuracy_from_files(starting_folder):
    results = []
    
    for root, dirs, files in os.walk(starting_folder):
        for file in glob.glob(os.path.join(root, 'log.txt')):
            with open(file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line.startswith('Accuracy'):
                        rel_path = os.path.relpath(file, starting_folder)
                        results.append((rel_path, last_line))
    return results

starting_folder = os.getcwd()
starting_folder = os.path.join(starting_folder, 'record')

log_files_with_accuracy = get_accuracy_from_files(starting_folder)
count = 0
dictionary_with_best_accuracy = {}
dictionary_overall_accuracy = {}

for file, accuracy in log_files_with_accuracy:
    print(f"{file}: {accuracy}")
    count += 1
    if count %3 == 0:
        print()

    file_ = file.split('\\')
    s = str(accuracy).split()
    acc = float(s[1])
    if file_[1] not in dictionary_with_best_accuracy:
        dictionary_with_best_accuracy[file_[1]] = (acc, file_[0])
    else:
        file_ = file.split('\\')
        if acc > dictionary_with_best_accuracy[file_[1]][0]:
            dictionary_with_best_accuracy[file_[1]] = (acc, file_[0])
    
    if file_[0] not in dictionary_overall_accuracy:
        dictionary_overall_accuracy[file_[0]] = acc
    else:
        dictionary_overall_accuracy[file_[0]] += acc
    
for key, value in dictionary_overall_accuracy.items():
    dictionary_overall_accuracy[key] = value / 3

print()
for key, value in dictionary_with_best_accuracy.items():
    print(f"{key}: {value}")
print()
for key, value in dictionary_overall_accuracy.items():
    if value > 60:
        print(f"{key}: {value:.2f}")