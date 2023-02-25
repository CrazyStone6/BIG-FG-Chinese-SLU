# -*- coding: utf-8 -*-



import torch
import torch.nn as nn

from transformers import get_linear_schedule_with_warmup
import os
import time
import math
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

from src.utils import computeF1Score, restore_order, save_slu_error_result, save_slu_result

class Processor(object):
    # TODO: divide original text into char-level text and word-level text, and provide alignment information between two level texts

    def __init__(self, dataset, model,optimizer,start_epoch, batch_size,args):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = batch_size
        self.start_epoch = start_epoch
        self.__args=args
        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            with open(os.path.join(self.__args.save_dir, "out.txt"), "a") as f:  # Open file for writing
                print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con),file=f)

        self.__criterion = nn.NLLLoss()

        t_total = math.ceil(self.__dataset.num_training_samples / self.__batch_size) * self.__args.num_epoch
        num_warmup_steps = math.ceil(t_total*0.1)

        self.__optimizer = optimizer
        self.__scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    def train(self):

        best_dev_metric = 0.0

        dataloader = self.__dataset.batch_delivery('train')
        for epoch in range(self.start_epoch+1, self.__args.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0

            time_start = time.time()
            self.__model.train()
            for batch in tqdm(dataloader, ncols=50) :
                # batch
                # 0: char_text_batch
                # 1: word_text_batch
                # 2: align_info_batch
                # 3: slot_batch
                # 4: intent_batch
                # 5: word_list_batch
                # 6: sent_seg_batch
                # 7: bert_input_id
                # 8: bert_attention_mask
                # 9: bert_token_type_id
                padded_char_text, padded_word_text, sorted_char_items, sorted_word_items, char_seq_lens, word_seq_lens, _ = \
                    self.__dataset.add_padding(
                        batch[0], batch[1],
                        char_items=[(batch[3], False), (batch[4], False)],
                        word_items=[(batch[2], False)]
                    )
                sorted_intent = list(Evaluator.expand_list(sorted_char_items[1]))

                char_text_var = torch.tensor(padded_char_text, dtype=torch.long)

                word_text_var = torch.tensor(padded_word_text, dtype=torch.long)
                slot_var = torch.tensor(list(Evaluator.expand_list(sorted_char_items[0])), dtype=torch.long)
                intent_var = torch.tensor(sorted_intent, dtype=torch.long)

                if torch.cuda.is_available():

                    char_text_var = char_text_var.cuda()

                    word_text_var = word_text_var.cuda()
                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                random_slot = random.random()
                slot_out, intent_out = \
                    self.__model(char_text_var, char_seq_lens,
                                 word_text_var, word_seq_lens, sorted_word_items[0],

                                )

                slot_loss = self.__criterion(slot_out, slot_var)
                intent_loss = self.__criterion(intent_out, intent_var)
                a = self.__args.alpha_rate
                batch_loss = a*slot_loss + (1-a)*intent_loss

                self.__optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.__model.parameters(), self.__dataset.max_grad_norm)
                self.__optimizer.step()
                self.__scheduler.step()

                try:
                    total_slot_loss += slot_loss.cpu().item()
                    total_intent_loss += intent_loss.cpu().item()
                except AttributeError:
                    total_slot_loss += slot_loss.cpu().data.numpy()[0]
                    total_intent_loss += intent_loss.cpu().data.numpy()[0]

            time_con = time.time() - time_start
            with open(os.path.join(self.__args.save_dir, "out.txt"), "a") as f:  # Open file for writing
                # print("Correlation is ", np.mean(rho), file=f)
                print('[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent data is {:2.6f}, '
                  'cost about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con),file=f)

            time_start = time.time()
            dev_f1_score, dev_acc, dev_sent_acc = self.estimate(if_dev=False, test_batch=self.__batch_size*2)

            # dev_metric = dev_f1_score
            dev_metric = dev_sent_acc
            with open(os.path.join(self.__args.save_dir, "out.txt"), "a") as f:  # Open file for writing
                # print("Correlation is ", np.mean(rho), file=f)
                print('\nTest result: slot f1 score: {:.6f}, intent acc score: {:.6f}, '
                      'semantic accuracy score: {:.6f}.'.format(dev_f1_score, dev_acc, dev_sent_acc), file=f)
            if dev_metric > best_dev_metric:
                test_f1, test_acc, test_sent_acc = self.estimate(if_dev=False, test_batch=self.__batch_size*2)

                best_dev_metric = dev_metric

                with open(os.path.join(self.__args.save_dir, "out.txt"), "a") as f:  # Open file for writing
                # print("Correlation is ", np.mean(rho), file=f)
                    print('\nTest_finally result: slot f1 score: {:.6f}, intent acc score: {:.6f}, '
                      'semantic accuracy score: {:.6f}.'.format(test_f1, test_acc, test_sent_acc),file=f)

                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    os.mkdir(model_save_dir)


                state = {'model': self.__model, 'dataset':self.__dataset,'optimizer': self.__optimizer, 'epoch': epoch}
                torch.save(state, os.path.join(model_save_dir, "MLWA_epoch.pkl"))

                time_con = time.time() - time_start
                with open(os.path.join(self.__args.save_dir, "out.txt"), "a") as f:  # Open file for writing
                # print("Correlation is ", np.mean(rho), file=f)
                    print('[Epoch {:2d}]: In test process, the slot f1 score is {:2.6f}, '
                      'the intent acc is {:2.6f}, the semantic acc is {:2.6f}, '
                      'cost about {:2.6f} seconds.\n'.format(epoch, dev_f1_score, dev_acc, dev_sent_acc, time_con),file=f)

    def estimate(self, if_dev, test_batch=100):
        """
        Estimate the performance of model on dev or test dataset.
        """

        if if_dev:
            pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "dev", test_batch
            )
        else:
            pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "test", test_batch
            )

        slot_f1 = computeF1Score(pred_slot, real_slot)[0]
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        return slot_f1, intent_acc, sent_acc

    @staticmethod
    def validate(model, dataset, batch_size):
        """
        validation will write mistaken samples to files and make scores.
        """
        pred_slot, real_slot, pred_intent, real_intent = Processor.prediction(model, dataset, "test", batch_size)

        slot_f1 = computeF1Score(pred_slot, real_slot)[0]
        intent_acc = Evaluator.accuracy(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)

        return slot_f1, intent_acc, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        char_text, word_text = [], []
        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []

        for batch in tqdm(dataloader, ncols=50) :
            # batch
            # 0: char_text_batch
            # 1: word_text_batch
            # 2: align_info_batch
            # 3: slot_batch
            # 4: intent_batch
            # 5: word_list_batch
            # 6: sent_seg_batch
            # 7: bert_input_id
            # 8: bert_attention_mask
            # 9: bert_token_type_id
            padded_char_text, padded_word_text, sorted_char_items, sorted_word_items, char_seq_lens, word_seq_lens, sorted_index = \
                dataset.add_padding(
                    batch[0], batch[1],
                    char_items=[(batch[3], False), (batch[4], False)],
                    word_items=[(batch[2], False)],
                    digital=False
                )
            sorted_intent = list(Evaluator.expand_list(sorted_char_items[1]))

            char_text.extend(restore_order(padded_char_text, sorted_index))
            word_text.extend(restore_order(padded_word_text, sorted_index))
            real_slot.extend(restore_order(sorted_char_items[0], sorted_index))
            real_intent.extend(restore_order(sorted_intent, sorted_index))


            digit_char_text = dataset.char_alphabet.get_index(padded_char_text)
            var_char_text = torch.tensor(digit_char_text, dtype=torch.long)

            digit_word_text = dataset.word_alphabet.get_index(padded_word_text)
            digit_intent = dataset.intent_alphabet.get_index(sorted_intent)
            var_word_text = torch.tensor(digit_word_text, dtype=torch.long)


            if torch.cuda.is_available():

                var_char_text = var_char_text.cuda()

                var_word_text = var_word_text.cuda()


            slot_idx, intent_idx = model(var_char_text, char_seq_lens,
                                         var_word_text, word_seq_lens, sorted_word_items[0]
                                          , n_predicts=1)

            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], char_seq_lens)[0]
            pred_slot.extend(restore_order(dataset.slot_alphabet.get_instance(nested_slot), sorted_index))
            pred_intent.extend(
                restore_order(dataset.intent_alphabet.get_instance(list(Evaluator.expand_list(intent_idx))),
                              sorted_index))

        # save error result
        save_slu_error_result(pred_slot, real_slot, pred_intent, real_intent, [char_text, word_text], dataset.save_dir, mode)
        # save slu result
        save_slu_result(pred_slot, real_slot, pred_intent, real_intent, [char_text, word_text], dataset.save_dir, mode)

        return pred_slot, real_slot, pred_intent, real_intent


class Evaluator(object):

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """

        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items