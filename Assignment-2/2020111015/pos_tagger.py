import torch
import numpy as np
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from conllu import parse
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import MulticlassF1Score as multiclass_f1_score
from arguments import get_args
import warnings
import os
from tqdm import tqdm
import seaborn as sns
from torchmetrics.classification import MulticlassConfusionMatrix
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt



args = get_args()

train_file = "./ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-train.conllu"
dev_file = "./ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-dev.conllu"
test_file = "./ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-test.conllu"

def read_conllu(file):
    with open(file, "r") as f:
        data = f.read()
    return parse(data)



def get_sentences(data):
    sentences = []
    for sentence in data:
        sentence = [word['form'] for word in sentence]
        sentences.append(sentence)
    return sentences


def get_tags_raw(data):
    tags = []
    for sentence in data:
        sentence = [word['upostag'] for word in sentence]
        tags.append(sentence)
    return tags

def get_tags(data):
    tags = []
    for sentence in data:
        sentence = [tags_to_num[word] for word in sentence]
        tags.append(sentence)
    return tags



train_data = read_conllu(train_file)
dev_data = read_conllu(dev_file)
test_data = read_conllu(test_file)

train_sentences = get_sentences(train_data)
dev_sentences = get_sentences(dev_data)
test_sentences = get_sentences(test_data)


train_tags = get_tags_raw(train_data)
dev_tags = get_tags_raw(dev_data)
test_tags = get_tags_raw(test_data)


tags_to_num = {}

for sentence in train_tags:
    for tag in sentence:
        tags_to_num[tag] = 1


for i, tags in enumerate(sorted(tags_to_num.keys())):
    tags_to_num[tags] = i



tags_to_num["PAD"] = len(tags_to_num)

# remove sentences with extra tags


indices = []

temp_sentences = []
temp_tags = []
for i in range(len(dev_sentences)):
    temp_sentences.append([])
    temp_tags.append([])
    for j in range(len(dev_tags[i])):
        if dev_tags[i][j] not in tags_to_num:
            continue
        temp_sentences[i].append(dev_sentences[i][j])
        temp_tags[i].append(dev_tags[i][j])

dev_sentences = temp_sentences
dev_tags = temp_tags



temp_sentences = []
temp_tags = []
for i in range(len(test_sentences)):
    temp_sentences.append([])
    temp_tags.append([])
    for j in range(len(test_tags[i])):
        if test_tags[i][j] not in tags_to_num:
            continue
        temp_sentences[i].append(test_sentences[i][j])
        temp_tags[i].append(test_tags[i][j])

test_sentences = temp_sentences
test_tags = temp_tags


train_tags = get_tags(train_tags)
dev_tags = get_tags(dev_tags)
test_tags = get_tags(test_tags)


num_to_tag = {v: k for k, v in tags_to_num.items()}





START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

class POSDataset1(Dataset):
    def __init__(self, p, s, data:list[tuple[list[str], list[int]]], vocabulary:Vocab|None=None):
        """Initialize the dataset. Setup Code goes here"""
        self.p = p
        self.s = s
        self.sentences = [i[0] for i in data]
        self.labels = [i[1] for i in data]


        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN]) # use min_freq for handling unkown words better
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            # if vocabulary provided use that
            self.vocabulary = vocabulary

        self.sentences = []
        self.labels = []
        for sentence, label in data:
            sentence = [START_TOKEN] + sentence + [END_TOKEN]
            label = [tags_to_num["PAD"]] + label + [tags_to_num["PAD"]]
            sentence = [PAD_TOKEN] * (self.p) + sentence + [PAD_TOKEN] * (self.s)
            label = [tags_to_num["PAD"]] * (self.p) + label + [tags_to_num["PAD"]] * (self.s)

            # split into p+s+1 chunks
            for i in range(self.p, len(sentence)-self.s):
                temp = sentence[i-self.p:i+self.s+1]
                self.sentences.append(temp)
                self.labels.append(torch.nn.functional.one_hot(torch.tensor(label[i]), num_classes=len(tags_to_num)))
        

    def __len__(self) -> int:
        """Returns number of datapoints."""
        return len(self.sentences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the datapoint at `index`."""
        return torch.tensor(self.vocabulary.lookup_indices(self.sentences[index])), self.labels[index]

    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of datapoints, batch them together"""
        sentences = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN]) # pad sentences with pad token id
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=torch.tensor(tags_to_num["PAD"])) # pad labels with 17

        return padded_sentences, padded_labels
    

class FNN_POS_Tagger(nn.Module):
    def __init__(self, p, s, vocabulary_size: int, config: int):
        super().__init__()
        self.p = p
        self.s = s
        
        if config == 1:
            """Embedding size: 64, 1 hidden layer of size 32, ReLU activation function."""
            self.embedding_module = torch.nn.Embedding(vocabulary_size, 64)
            self.fnn = torch.nn.Sequential(
                                        torch.nn.Linear(64, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(0.2),
                                        torch.nn.Linear(32, len(tags_to_num)))
        elif config == 2:
            """Embedding size: 128, 1 hidden layer of size 64, LeakyReLU activation function."""
            self.embedding_module = torch.nn.Embedding(vocabulary_size, 128)
            self.fnn = torch.nn.Sequential(
                                        torch.nn.Linear(128, 64),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.Dropout(0.2),
                                        torch.nn.Linear(64, len(tags_to_num)))
            
        elif config == 3:
            """Embedding size: 128, 2 hidden layers of size 64 and 64, ReLU activation function."""
            self.embedding_module = torch.nn.Embedding(vocabulary_size, 128)
            self.fnn = torch.nn.Sequential(
                                        torch.nn.Linear(128, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(0.2),
                                        torch.nn.Linear(64, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(0.2),
                                        torch.nn.Linear(32, len(tags_to_num)))
            


    def forward(self, x):
        x = self.embedding_module(x)
        x = self.fnn(x)

        # x = self.p i self.s
        x = x[:, self.p, :]
        return x



def train(model, train_loader, device, dev_loader, dev_dataset, criterion, optimizer):
    loss_list = []
    accuracy_list = []
    f1_list = []

    total_dev_labels = []
    total_dev_predictions = []
    for epoch in tqdm(range(10), desc="Training", unit="epoch"):
        for step, (word, tag) in enumerate(train_loader):
            word, tag = word.to(device), tag.to(device)
            optimizer.zero_grad()
            output = model(word)
            tag = tag.float()
            loss = criterion(output, tag)
            loss.backward()
            optimizer.step()

            if step%1000 == 0:
                loss_list.append(loss.item())

            #     print(f"Epoch {epoch} Step {step} Loss: {loss.item():.3f}")

        correct = 0
        total = 0
        with torch.no_grad():
            dev_predictions = []
            dev_labels = []
            for word, tag in dev_loader:
                word, tag = word.to(device), tag.to(device)
                output = model(word)
                output = torch.argmax(output, dim=1)
                tag = torch.argmax(tag, dim=1)
                
                correct += (output == tag).sum()
                dev_predictions.extend(output.tolist())
                dev_labels.extend(tag.tolist())

            # caclulate f1 score
            metric = multiclass_f1_score(num_classes=len(tags_to_num), average='macro')
            dev_f1_score = metric(torch.tensor(dev_predictions), torch.tensor(dev_labels))

            total_dev_labels.extend(dev_labels)
            total_dev_predictions.extend(dev_predictions)            

        # print()
        # print(f"Epoch {epoch} Accuracy: {correct/len(dev_dataset):.3f} Dev F1 Score: {dev_f1_score:.3f}")
        # print()
        accuracy_list.append(correct/len(dev_dataset))
        f1_list.append(dev_f1_score)
    cm = MulticlassConfusionMatrix(num_classes=len(tags_to_num))
    cm(torch.tensor(total_dev_predictions), torch.tensor(total_dev_labels))

    return loss_list, accuracy_list, f1_list, cm
        

def test(model, test_loader, test_dataset, device):
    correct = 0
    total = 0
    with torch.no_grad(): 
        test_predictions = []
        test_labels = []
        for word, tag in test_loader:
            word, tag = word.to(device), tag.to(device)
            output = model(word)
            output = torch.argmax(output, dim=1)
            tag = torch.argmax(tag, dim=1)
            correct += (output == tag).sum()
            test_predictions.extend(output.tolist())
            test_labels.extend(tag.tolist())

        # caclulate f1 score
        metric = multiclass_f1_score(num_classes=len(tags_to_num), average='macro')
        test_f1_score = metric(torch.tensor(test_predictions), torch.tensor(test_labels))
        cm = MulticlassConfusionMatrix(num_classes=len(tags_to_num))
        cm(torch.tensor(test_predictions), torch.tensor(test_labels))

    print()
    print(f"Test Accuracy: {correct/len(test_dataset):.3f} Test F1 Score: {test_f1_score:.3f}")

    return cm


START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

class POSDataset2(Dataset):
  def __init__(self, data: list[tuple[list[str], list[int]]], vocabulary:Vocab|None=None):
    """Initialize the dataset. Setup Code goes here"""
    self.sentences = [i[0] for i in data]
    self.labels = [i[1] for i in data]


    if vocabulary is None:
      self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN]) # use min_freq for handling unkown words better
      self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
    else:
      # if vocabulary provided use that
      self.vocabulary = vocabulary

    self.sentences = []
    self.labels = []
    for j,(sentence, label) in enumerate(data):
      sentence = [START_TOKEN] + sentence + [END_TOKEN]
      label = [tags_to_num["PAD"]] + label + [tags_to_num["PAD"]]

      # split into p+s+1 chunks
      self.sentences.append(sentence)
      self.labels.append(torch.nn.functional.one_hot(torch.tensor(label), num_classes=len(tags_to_num)))


  def __len__(self) -> int:
    """Returns number of datapoints."""
    return len(self.sentences)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the datapoint at `index`."""
    return torch.tensor(self.vocabulary.lookup_indices(self.sentences[index])), torch.tensor(self.labels[index])

  def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a list of datapoints, batch them together"""
    sentences = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN]) # pad sentences with pad token id
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=torch.tensor(tags_to_num["PAD"])) # pad labels with 17

    return padded_sentences, padded_labels


class RNN_POS_Tagger(nn.Module):
    def __init__(self, vocabulary_size: int, config: int):
        super().__init__()

        if config == 1:
            """Embedding size: 128, 1 LSTM layer of size 128, 1 linear layer. No Dropout"""
            self.embedding = nn.Embedding(vocabulary_size, 128)
            self.lstm = nn.LSTM(128, 128, batch_first=True)
            self.linear = nn.Sequential(
                nn.Linear(128, len(tags_to_num)),
                nn.LogSoftmax(dim=2)
            )

        elif config == 2:
            """Embedding size: 128, 1 LSTM layer of size 128, 2 linear layers."""
            self.embedding = nn.Embedding(vocabulary_size, 128)
            self.lstm = nn.LSTM(128, 128, batch_first=True)
            self.linear = nn.Sequential(
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                torch.nn.Dropout(0.2),
                nn.Linear(64, len(tags_to_num)),
                nn.LogSoftmax(dim=2)
            )

        elif config == 3:
            """Embedding size: 128, 2 LSTM layers of size 128, 2 linear layers."""
            self.embedding = nn.Embedding(vocabulary_size, 128)
            self.lstm = nn.LSTM(128, 128, batch_first=True, num_layers=2)
            self.linear = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                torch.nn.Dropout(0.2),
                nn.Linear(64, len(tags_to_num)),
                nn.LogSoftmax(dim=2)
            )

        elif config == 4:
            """Embedding size: 128, 1 LSTM biderectional layer of size 128, 2 linear layer"""
            self.embedding = nn.Embedding(vocabulary_size, 128)
            self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
            self.linear = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                torch.nn.Dropout(0.2),
                nn.Linear(64, len(tags_to_num)),
                nn.LogSoftmax(dim=2)
            )

        elif config == 5:
            """Embedding size: 32, 1 LSTM layer of size 32, 1 linear layer. No Dropout"""
            self.embedding = nn.Embedding(vocabulary_size, 32)
            self.lstm = nn.LSTM(32, 32, batch_first=True)
            self.linear = nn.Sequential(
                nn.Linear(32, len(tags_to_num)),
                nn.LogSoftmax(dim=2)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def train_rnn(model, train_loader, train_dataset, dev_loader, dev_dataset, criterion, optimizer, device):
    loss_list = []
    accuracy_list = []
    f1_list = []
    total_dev_labels = []
    total_dev_predictions = []

    for epoch in tqdm(range(10), desc="Training", unit="epoch"):
        for step, (word, tag) in enumerate(train_loader):
            word, tag = word.to(device), tag.to(device)
            optimizer.zero_grad()
            output = model(word)
            tag = tag.float()
            loss = 0
            for i in range(tag.shape[1]):
                loss += criterion(output[:,i,:], torch.argmax(tag[:,i,:], dim=1))

            loss.backward()
            optimizer.step()

            if step%1000 == 0:
                loss_list.append(loss.item())

            #     print(f"Epoch {epoch} Step {step} Loss: {loss.item():.3f}")

        correct = 0
        total = 0
        with torch.no_grad():
            dev_predictions = []
            dev_labels = []
            for word, tag in dev_loader:
                word, tag = word.to(device), tag.to(device)
                output = model(word)
                output = torch.argmax(output, dim=2)
                tag = torch.argmax(tag, dim=2)
                for i in range(tag.shape[1]):
                    correct += (output[:,i] == tag[:,i]).sum()
                    dev_predictions.extend(output[:,i].tolist())
                    dev_labels.extend(tag[:,i].tolist())
                    total += tag.shape[0]

            # caclulate f1 score
            metric = multiclass_f1_score( num_classes=len(tags_to_num), average='macro')
            dev_f1_score = metric(torch.tensor(dev_predictions), torch.tensor(dev_labels))
            total_dev_labels.extend(dev_labels)
            total_dev_predictions.extend(dev_predictions)


        # print()
        # print(f"Epoch {epoch} Accuracy: {correct/total:.3f} Dev F1 Score: {dev_f1_score:.3f}")
        # print()
        accuracy_list.append(correct/total)
        f1_list.append(dev_f1_score)

    cm = MulticlassConfusionMatrix(num_classes=len(tags_to_num))
    cm(torch.tensor(total_dev_predictions), torch.tensor(total_dev_labels))

    return loss_list, accuracy_list, f1_list, cm


def test_rnn(model, test_loader, device, ):
    correct = 0
    total = 0
    with torch.no_grad(): 
        test_predictions = []
        test_labels = []
        for word, tag in test_loader:
            word, tag = word.to(device), tag.to(device)
            output = model(word)
            output = torch.argmax(output, dim=2)
            tag = torch.argmax(tag, dim=2)
            for i in range(tag.shape[1]):
                correct += (output[:,i] == tag[:,i]).sum()
                test_predictions.extend(output[:,i].tolist())
                test_labels.extend(tag[:,i].tolist())
                total += tag.shape[0]

        # caclulate f1 score
        metric = multiclass_f1_score(num_classes=len(tags_to_num), average='macro')
        test_f1_score = metric(torch.tensor(test_predictions), torch.tensor(test_labels))

    cm = MulticlassConfusionMatrix(num_classes=len(tags_to_num))
    cm(torch.tensor(test_predictions), torch.tensor(test_labels))


    print()
    print(f"Test Accuracy: {correct/total:.3f} Test F1 Score: {test_f1_score:.3f}")

    return cm

def save_model(model, path):
    torch.save(model.state_dict(), path)

def plot_graph(data, type_n, name):

    # convert data to numpy array
    plt.plot(data)
    plt.xlabel("Epoch")
    plt.ylabel(type_n)
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fnn_model = None
    rnn = None


    if args.type == "f":
        p, s = 3, 2

        train_dataset = POSDataset1(p,s, list(zip(train_sentences, train_tags)))
        dev_dataset = POSDataset1(p,s,list(zip(dev_sentences, dev_tags)), vocabulary=train_dataset.vocabulary)
        test_dataset = POSDataset1(p,s,list(zip(test_sentences, test_tags)), vocabulary=train_dataset.vocabulary)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=32)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        models = []

        criterion = torch.nn.CrossEntropyLoss() # use ignore index to ignore losses for padding value indices
        cond = 0
        for config in range(1, 4):
            if cond == 1:
                break
            for p, s in [(0,0), (1,1), (2,2), (3,3), (4,4)]:
                if os.path.exists(f"fnn_model_{config}_{p}_{s}.pth"):
                    fnn_model = FNN_POS_Tagger(p, s, len(train_dataset.vocabulary), config)
                    fnn_model.load_state_dict(torch.load(f"fnn_model_{config}_{p}_{s}.pth"))
                    fnn_model.to(device)
                    cond = 1
                    break
        for config in range(1, 4):
            if cond == 1:
                break
            for p, s in [(0,0), (1,1), (2,2), (3,3), (4,4)]:
                fnn_model = FNN_POS_Tagger(p, s, len(train_dataset.vocabulary), config)
                optimizer = torch.optim.SGD(fnn_model.parameters(), lr=1e-1)
                # fnn_model = torch.nn.DataParallel(fnn_model)
                fnn_model.to(device)

                if not os.path.exists(f"fnn_model_{config}_{p}_{s}.pth"):
                    loss_list, accuracy_list, f1_list, cm = train(fnn_model, train_loader, device, dev_loader, dev_dataset, criterion, optimizer)
                    accuracy_list = [_.detach().cpu().numpy() for _ in accuracy_list]
                    f1_list = [_.detach().cpu().numpy() for _ in f1_list]
                    final_acc = accuracy_list[-1]
                    models.append((fnn_model, config, p, s, final_acc))
                    # make and save plots
                    plot_graph(loss_list, "Loss", f"results/fnn_loss_{config}_{p}_{s}.png")
                    plot_graph(accuracy_list, "Accuracy",  f"results/fnn_accuracy_{config}_{p}_{s}.png")
                    plot_graph(f1_list, "F1 Score",  f"results/fnn_f1_{config}_{p}_{s}.png")
                    sns.heatmap(cm.compute().detach().cpu().numpy(), annot=True, fmt="d", xticklabels=[num_to_tag[i] for i in range(len(tags_to_num))], yticklabels=[num_to_tag[i] for i in range(len(tags_to_num))] )
                    plt.xticks(rotation=45)
                    plt.xlabel("Predicted")
                    plt.ylabel("Ground Truth")
                    plt.yticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f"results/fnn_cm_{config}_{p}_{s}.png")
                    plt.close()


        if cond == 0:

            best_model = None
            best_acc = 0
            for model, config, p, s, acc in models:
                if acc > best_acc:
                    best_model = (model, config, p, s)
                    best_acc = acc

            print(f"Best Model Accuracy: {best_acc} Config: {best_model[1]} p: {best_model[2]} s: {best_model[3]}")
            cm = test(best_model[0], test_loader, test_dataset, device)
            sns.heatmap(cm.compute().detach().cpu().numpy(), annot=True, fmt="d", xticklabels=[num_to_tag[i] for i in range(len(tags_to_num))], yticklabels=[num_to_tag[i] for i in range(len(tags_to_num))] )
            plt.xticks(rotation=45)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"results/fnn_cm_{best_model[1]}_{best_model[2]}_{best_model[3]}_test.png")
            plt.close()
            save_model(best_model[0], f"fnn_model_{best_model[1]}_{best_model[2]}_{best_model[3]}.pth")
            fnn_model = best_model[0]
            p = best_model[2]
            s = best_model[3]

                    

    elif args.type == "r":

        train_dataset = POSDataset2( list(zip(train_sentences, train_tags)))
        dev_dataset = POSDataset2(list(zip(dev_sentences, dev_tags)), vocabulary=train_dataset.vocabulary)
        test_dataset = POSDataset2(list(zip(test_sentences, test_tags)), vocabulary=train_dataset.vocabulary)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=train_dataset.collate, shuffle=True)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=32, collate_fn=dev_dataset.collate)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, collate_fn=test_dataset.collate)
       

        models = []
        cond = 0
        for config in range(1, 6):
            if os.path.exists(f"rnn_model_{config}.pth"):
                cond = 1
                break
        for config in range(1, 6):
            rnn = RNN_POS_Tagger(len(train_dataset.vocabulary), config)
            if os.path.exists(f"rnn_model_{config}.pth"):
                rnn.load_state_dict(torch.load(f"rnn_model_{config}.pth"))
                rnn.to(device)

                cond = 1
                break


            criterion = nn.NLLLoss()
            optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

            rnn.to(device)

            if cond == 0:
                loss_list, accuracy_list, f1_list, cm = train_rnn(rnn, train_loader, train_dataset, dev_loader, dev_dataset, criterion, optimizer, device)
                accuracy_list = [_.detach().cpu().numpy() for _ in accuracy_list]
                f1_list = [_.detach().cpu().numpy() for _ in f1_list]
                final_acc = accuracy_list[-1]
                models.append((rnn, config, final_acc))
                # make and save plots
                plot_graph(loss_list, "Loss", f"results/rnn_loss_{config}.png")
                plot_graph(accuracy_list, "Accuracy",  f"results/rnn_accuracy_{config}.png")
                plot_graph(f1_list, "F1 Score",  f"results/rnn_f1_{config}.png")
                sns.heatmap(cm.compute().detach().cpu().numpy(), annot=True, fmt="d", xticklabels=[num_to_tag[i] for i in range(len(tags_to_num))], yticklabels=[num_to_tag[i] for i in range(len(tags_to_num))] )
                plt.xticks(rotation=45)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.yticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"results/rnn_cm_{config}.png")
                plt.close()


        if cond == 0:
            best_model = None
            best_acc = 0
            for model, config, acc in models:
                if acc > best_acc:
                    best_model = (model, config)
                    best_acc = acc

            print(f"Best Model Accuracy: {best_acc} Config: {best_model[1]}")
            cm = test_rnn(best_model[0], test_loader, device)
            sns.heatmap(cm.compute().detach().cpu().numpy(), annot=True, fmt="d", xticklabels=[num_to_tag[i] for i in range(len(tags_to_num))], yticklabels=[num_to_tag[i] for i in range(len(tags_to_num))] )
            plt.xticks(rotation=45)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.yticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"results/rnn_cm_{best_model[1]}_test.png")
            plt.close()
            save_model(best_model[0], f"rnn_model_{best_model[1]}.pth")
            rnn = best_model[0]
            
            

    if args.type == "f":
        sentence = input("Enter a sentence: ")   
        sentence = sentence.lower()
        # remove punctuations
        sentence = sentence.replace(".", "")
        sentence = sentence.replace(",", "")
        sentence = sentence.replace("?", "")
        sentence = sentence.replace("!", "")
        sentence = sentence.replace(";", "")
        sentence = sentence.replace(":", "")

        sentence = word_tokenize(sentence)

        sentence = [START_TOKEN] + sentence + [END_TOKEN]
        sentence = [PAD_TOKEN] * p + sentence + [PAD_TOKEN] * s

        sentence_cp = sentence
        sentence = torch.tensor(train_dataset.vocabulary.lookup_indices(sentence)).to(device)
        # split into chunks of p+s+1
        chunks = [sentence[i:i+p+s+1] for i in range(len(sentence)-p-s)]
        chunks_cp = [sentence_cp[i:i+p+s+1] for i in range(len(sentence_cp)-p-s)]


        for i, chunk in enumerate(chunks):
            output = fnn_model(chunk.unsqueeze(0))
            output = torch.argmax(output, dim=1)
            print(f"{chunks_cp[i][fnn_model.s]}\t{num_to_tag[output.item()]}")

    elif args.type == "r":
        sentence = input("Enter a sentence: ")
        sentence = sentence.lower()
        # remove punctuations
        sentence = sentence.replace(".", "")
        sentence = sentence.replace(",", "")
        sentence = sentence.replace("?", "")
        sentence = sentence.replace("!", "")
        sentence = sentence.replace(";", "")
        sentence = sentence.replace(":", "")
        sentence = word_tokenize(sentence)
        sentence = [START_TOKEN] + sentence + [END_TOKEN]
        sentence_cp = sentence
        sentence = torch.tensor(train_dataset.vocabulary.lookup_indices(sentence)).to(device)
        # split into chunks of p+s+1
        output = rnn(sentence.unsqueeze(0))
        output = torch.argmax(output, dim=2)
        for i in range(output.shape[1]):
            print(f"{sentence_cp[i]}\t{num_to_tag[output[0,i].item()]}")
