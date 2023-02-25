# -*- coding: utf-8 -*-



from src.modules.module import ModelManager

from src.data_loader.loader import DatasetManager
from src.process import Processor
import torch.optim as optim
import torch

import os
import json
import random
import argparse
import numpy as np


parser = argparse.ArgumentParser()

# Training parameters.
# TODO:
parser.add_argument('--do_evaluation', '-eval', action="store_true", default=False)

parser.add_argument('--data_dir', '-dd', type=str, default='data/cais_bies-True_token-hanlp')
parser.add_argument('--train_file_name', '-train_file', type=str, default='train.txt')
parser.add_argument('--valid_file_name', '-valid_file', type=str, default='dev.txt')
parser.add_argument('--test_file_name', '-test_file', type=str, default='test.txt')
parser.add_argument('--save_dir', '-sd', type=str, default='save')
parser.add_argument("--random_state", '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=1000)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.0005)
parser.add_argument("--max_grad_norm", "-mgn", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.5)
parser.add_argument('--alpha_rate', '-ar', type=float, default=0.7)


# model parameters.
# TODO:
parser.add_argument('--char_embedding_dim', '-ced', type=int, default=64)
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--char_attention_hidden_dim', '-cahd', type=int, default=1024)
parser.add_argument('--word_attention_hidden_dim', '-wahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=16)
parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=128)
parser.add_argument("--n_layers", '-nld', help='BIG layers number of decoder', type=int, default=2)
premodel = r"MLWA_epoch.pkl"
model_file_path = os.path.join(r"save/model",premodel)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.do_evaluation:
        # Save training and model parameters.
        if not os.path.exists(args.save_dir):
            os.system("mkdir -p " + args.save_dir)

        log_path = os.path.join(args.save_dir, "param.json")
        with open(log_path, "w") as fw:
            fw.write(json.dumps(args.__dict__, indent=True))
        # Fix the random seed of package random.
        random.seed(args.random_state)
        np.random.seed(args.random_state)
        # Fix the random seed of Pytorch when using GPU.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_state)
            torch.cuda.manual_seed(args.random_state)

        # Fix the random seed of Pytorch when using CPU.
        torch.manual_seed(args.random_state)
        torch.random.manual_seed(args.random_state)
        #加载预训练模型
        if os.path.exists(model_file_path):
            checkpoint = torch.load(model_file_path, map_location=device)
            model = checkpoint['model']
            dataset = checkpoint["dataset"]
            optimizer = checkpoint["optimizer"]
            start_epoch = checkpoint["epoch"]
            dataset.show_summary()
            model.show_summary()
            process = Processor(dataset, model,optimizer,start_epoch,args.batch_size,args)
            print('Load epoch {} weights suceess！'.format(start_epoch))
        else:
            # Instantiate a dataset object.
            print('No save model, will train from scratch！')
            start_epoch = 0
            dataset = DatasetManager(args)
            dataset.quick_build()
            dataset.show_summary()
            model_fn = ModelManager
            # Instantiate a network model object.
            model = model_fn(
                args, len(dataset.char_alphabet),
                len(dataset.word_alphabet),
                len(dataset.slot_alphabet),
                len(dataset.intent_alphabet)
            )
            model.show_summary()
            optimizer = optim.Adam(model.parameters(), lr=dataset.learning_rate,weight_decay=dataset.l2_penalty )

            # To train and evaluat e the models.
            process = Processor(dataset, model, optimizer,start_epoch,args.batch_size,args)
        try:
            process.train()
        except KeyboardInterrupt:
            print ("Exiting from training early.")

    checkpoint = torch.load(os.path.join(args.save_dir, "model/MLWA_epoch.pkl"),map_location=device)
    model = checkpoint['model']
    dataset = checkpoint["dataset"]

    print('\nAccepted performance: ' + str(Processor.validate(
        model, dataset, args.batch_size * 2)) + " at test dataset;\n")
