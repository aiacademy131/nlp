from __future__ import unicode_literals, print_function, division
from io import open
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import prepareData, trainIters, evaluate, evaluateRandomly 
from modules import device, MAX_LENGTH

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

################################################################################################################################

input_lang, output_lang, pairs = prepareData('eng', 'kor', True)
print(random.choice(pairs))

################################################################################################################################

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

################################################################################################################################

font_path = r'font/?????????.ttf' # ?????? ?????? ??????
font_name = fm.FontProperties(fname=font_path, size=50).get_name()
plt.rc('font', family=font_name)
# print(font_name)
# import matplotlib as mpl
# mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 15

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # ???????????? ????????? ??? locator??? tick??? ??????
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

################################################################################################################################

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

plot_losses = trainIters(encoder1, attn_decoder1, 75000, input_lang, output_lang, pairs, print_every=5000) # 75000
showPlot(plot_losses)

torch.save(encoder1, 'EncoderRNN.pth')
torch.save(attn_decoder1, 'AttnDecoderRNN.pth')

encoder1 = torch.load('EncoderRNN.pth')
attn_decoder1 = torch.load('AttnDecoderRNN.pth')

evaluateRandomly(encoder1, attn_decoder1, pairs, input_lang, output_lang, device)

################################################################################################################################

import unicodedata

# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "?????? ???????????????????????? ??????????????? ?????????????????? ???????????? ??????????????? .", input_lang, output_lang, device)
# plt.matshow(attentions.numpy())
# plt.show()

def showAttention(input_sentence, output_words, attentions):
    input_sentence = unicodedata.normalize('NFC',input_sentence) # ?????? ?????? ??????
    
    # colorbar??? ?????? ??????
    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    
    # ??? ??????
    ax.set_xticklabels([])
    # ax.set_xticklabels([''] + input_sentence.split(' ') +
    #                    ['<EOS>'], rotation=90, fontproperties=font_name)
    ax.set_yticklabels([''] + output_words, fontproperties=font_name)

    for i, word in enumerate(input_sentence.split(' ')):
        ax.text(-0.2+i*1, -0.8, word, rotation=90)   
    # for i, word in enumerate(output_words):
    #     ax.text(-0.8, i*1, word, horizontalalignment='right', size=15)

    # ??? ????????? ?????? ????????????
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # ticks_loc_x = ax.get_xticks().tolist()
    # ticks_loc_y = ax.get_yticks().tolist()
    # ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc_x))
    # ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc_y))
    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence, input_lang, output_lang, device)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


import warnings
warnings.filterwarnings("ignore")

evaluateAndShowAttention("??????????????? ??????????????? ?????????????????? ??????????????? ??????????????????????????? .")

evaluateAndShowAttention("?????? ???????????????????????? ????????????????????? ??????????????????????????? ??????????????? ??????????????? .")

evaluateAndShowAttention("????????? ?????? ????????????????????? ???????????????????????? ???????????? ???????????????????????? .")

evaluateAndShowAttention("??????????????? ?????????????????????????????? ??????????????? ?????????????????? ???????????????????????? .")


while True:
    a = input()
    if a in ["break", "exit"]: break
    evaluateAndShowAttention(random.choice(pairs)[0])