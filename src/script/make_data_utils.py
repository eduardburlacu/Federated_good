import csv
import os
import json
import torch
from src import PATH

def make_data(in_dir, out_dir, is_embedded=False):
    try:
        os.mkdir(out_dir)
    except FileExistsError as _:
        pass
    data_files = filter(lambda f: f.endswith(".json"), os.listdir(in_dir))
    dir_ctr = 0
    file_ctr = 0
    with open(os.path.join(out_dir, 'data.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=' ')
        if is_embedded:
            csv_writer.writerow(['img_name', 'label', 'client', 'img'])
        else:
            csv_writer.writerow(['img_name', 'label', 'client'])
        for data_file in data_files:
            print("path: ", os.path.join(in_dir, data_file))
            with open(os.path.join(in_dir, data_file), 'r') as f:
                data = json.load(f)
                for user, userdata in data['user_data'].items():
                    for x, y in zip(userdata['x'], userdata['y']):
                        file_ctr += 1
                        if file_ctr % 1000 == 0:
                            dir_ctr += 1
                            print(f"Processing file {file_ctr}.")
                        filename = str(file_ctr).zfill(7) + '.pt'
                        dirname = str(dir_ctr).zfill(3)
                        try:
                            os.mkdir(os.path.join(out_dir, dirname))
                        except FileExistsError as _:
                            pass
                        filepath = os.path.join(out_dir, dirname, filename)
                        torch.save(x, filepath)
                        if is_embedded:
                            csv_writer.writerow([os.path.join(dirname, filename), y, user, x])
                        else:
                            csv_writer.writerow([os.path.join(dirname, filename), y, user])

def get_latest_experiment_dir(eid):
    experiment_base_dir = os.path.join(os.path.join(PATH['config'],'simulations'), str(eid))
    experiment_dir = os.path.join(experiment_base_dir, max(os.listdir(experiment_base_dir)))
    return experiment_dir

def get_log_filename(eid):
    return os.path.join(get_latest_experiment_dir(eid), f"{eid}.log")

def prep_experiment_dir(eid):
    experiment_base_dir = os.path.join(os.path.join(PATH['config'],'simulations'), str(eid))
    try:
        os.mkdir(experiment_base_dir)
    except FileExistsError as _:
        pass
    try:
        experiment_idx = max([int(max(n.lstrip("0"), "0")) for n in os.listdir(experiment_base_dir)]) + 1
    except ValueError as _:
        experiment_idx = 0
    experiment_dir = os.path.join(experiment_base_dir, str(experiment_idx).zfill(3))
    try:
        os.mkdir(experiment_dir)
    except FileExistsError as _:
        pass
    return experiment_dir