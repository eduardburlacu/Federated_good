# Based on FedProx/data/synthetic_0.5_0.5/generate_synthetic.py

import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math

os.chdir(os.path.dirname(__file__))

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, beta, min_samples, max_samples, iid, num_users):
    print("alpha, beta, min_samples, max_samples, iid, num_users", alpha, beta, min_samples, max_samples, iid, num_users)
    dimension = 60
    NUM_CLASS = 10
    
    num_samples = 0
    while num_samples < min_samples or num_samples > max_samples:
        # print(f"num_samples = {num_samples}, repeating.")
        samples_per_user = np.random.lognormal(4, 2, (num_users)).astype(int) + 50
        # print(samples_per_user)
        num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(num_users)]
    y_split = [[] for _ in range(num_users)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, num_users)
    mean_b = mean_W
    B = np.random.normal(0, beta, num_users)
    mean_x = np.zeros((num_users, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(num_users):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        # print(f"mean_x[{i}] {mean_x[i]}")

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(num_users):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))


    return X_split, y_split



def main():
    alpha, beta, min_samples, max_samples, name = list(map(int, sys.argv[1:6]))
    num_users = int(sys.argv[6]) if len(sys.argv) == 7 else 30

    np.random.seed(name)
    random.seed(name)

    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    if alpha == -1:
        X, y = generate_synthetic(
            alpha=0, beta=0, iid=1,
            min_samples=min_samples,
            max_samples=max_samples,
            num_users=num_users,
        )      # synthetic_IID
    elif alpha >= 0 and beta >= 0:
        X, y = generate_synthetic(
                alpha=alpha, beta=beta, iid=0,
                min_samples=min_samples,
                max_samples=max_samples,
                num_users=num_users,
        )
    else:
        print(f"Cannot handle alpha, beta {alpha}, {beta}. Exiting.")
        sys.exit(1)

    DATA_DIR = f"data_{alpha}_{beta}_{name}"
    try:
        os.mkdir(DATA_DIR)
    except FileExistsError as _:
        pass

    train_path = os.path.join(DATA_DIR, "mytrain.json")
    test_path = os.path.join(DATA_DIR, "mytest.json")

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in trange(num_users, ncols=120):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)
    

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()

