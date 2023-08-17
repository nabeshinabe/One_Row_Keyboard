import os
import numpy as np
import random
import torch
import torch.nn as nn
import glob
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import Model

from pdb import set_trace

##### GPU・モデル保存設定 #####
torch.manual_seed(24)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = Path("./models/model.pt")
PATH = Path("./models/checkpoint.pt")

# writer_path = "./logs/logs"
# writer = SummaryWriter(log_dir=writer_path)

print("GPU:", torch.cuda.is_available())

print("Let's use", torch.cuda.device_count(), "GPUs!")

##### データ処理 #####
# Fileの読み込み
INPUT_DATA = "./Data/OneRowData"
TRUTH_DATA = "./Data/ConvertData"

input_files = glob.glob(f"{INPUT_DATA}/*[0-1]_onerow.txt")
print(input_files)

input_data = []
truth_data = []

# データセット作成
dataset = []

for inp in input_files:
    words_input_by_character = []
    words_truth_by_character = []

    with open(inp, 'r') as f:
        file_name = os.path.basename(inp)
        file_id = file_name.replace("_onerow.txt", "")

        content_input = f.read()  # ファイル内容を読み込む
        words_input = content_input.split()  # スペースで区切られた単語ごとに分割する

        # 各単語を文字区切りする
        for word in words_input:
            word_list = []

            for ch in word:
                chr_int = int(ch)

                onehot = torch.zeros(10)
                onehot[chr_int] = 1 # 文字ではないAsciiの部分は使用しない

                word_list.append(onehot)
            
            # 文字をtensor型に変換させて格納
            words_input_by_character.append(word_list)
            # words_input_by_character.append(torch.tensor(list(word)))

        # print(words_input_by_character)
    
    # 上記と同じファイル番号をもつinputを使用
    with open(f"{TRUTH_DATA}/{file_id}_base.txt", 'r') as f:
        content_base = f.read()  # ファイル内容を読み込む
        words_base = content_base.split()  # スペースで区切られた単語ごとに分割する

        # 各単語を文字区切りする
        for word in words_base:
            word_list = []

            for ch in word:
                # 各文字をasciiに変換
                chr_int = ord(ch)
                word_list.extend([chr_int])
            
            # 文字をtensor型に変換させて格納
            words_truth_by_character.append(torch.tensor(word_list))
            # words_input_by_character.append(torch.tensor(list(word)))

        # 両方あってペアデータ作成
        dataset.append([words_input_by_character, words_truth_by_character])
        # print(dataset)

##### Training Data と Test Dataの分割 #####
random.shuffle(dataset)

# 分割比率を設定する（ここでは 80% のデータを訓練データとし、残りの 20% をテストデータとします）
train_ratio = 0.8
train_size = int(len(dataset) * train_ratio)

# データを訓練データとテストデータに分割する
train_data = dataset[:train_size]
test_data = dataset[train_size:]

# 学習データと正解データの定義
# training_data = torch.tensor([6, 5, 2, 3, 5])
# target_data = torch.tensor([ord('B'),ord('A'),ord('H'),ord('I'),ord('A')])


#### Asciiの変換ルールを定義 ###
# 小文字しか扱わない

def int_to_ascii(num):
    # 33(!)→0 ,,, 64(@)→31
    # 91([)→32 ,,, 126(~)→67
    return_num = 65
    if 33 <= num and num <= 64:
        return_num = num - 33
    elif 91 <= num and num <= 126:
        return_num = num - 59
    return return_num

def ascii_to_int(num):
    # 0→33(!) ,,, 31→64(@)
    # 32→91([) ,,, 67→126(~)
    return_num = 65
    if 0 <= num and num <= 31:
        return_num = num + 33
    elif 32 <= num and num <= 67:
        return_num = num + 59
    return return_num



##### ニューラルネットワークの定義 #####
# ハイパーパラメータの設定
input_size = 10
output_size = 67

num_epochs = 20 # エポック数定義
batch_size = 4 # バッチサイズ定義
learning_rate = 0.0001 # 学習率


# データセットとデータローダーの作成
train_dataset = torch.empty(0)
target_dataset = torch.empty(0)


for dset in train_data:
    train = dset[0]
    target = dset[1]

    for (x_data, t_data) in zip(train, target):
        # Train Dataの作成
        x = torch.cat(x_data, dim=0).reshape(len(x_data), -1)
        train_dataset = torch.cat([train_dataset, x], dim=0)

        # Target Dataを作成
        t = torch.zeros(len(t_data), output_size)
        for i, t_num in enumerate(t_data):
            t[i][int_to_ascii(t_num)] = 1 # 文字ではないAsciiの部分は使用しない
        
        target_dataset = torch.cat([target_dataset, t], dim=0)

tensor_dataset = TensorDataset(train_dataset, target_dataset)
dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

model = Model.Model().to(device)

# 損失関数と最適化手法の定義
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones([output_size]))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for inputs, label in dataloader:
        # Forward Path
        outputs = model(inputs[0].float())

        loss = criterion(outputs, label[0])

        # バックワードパスとパラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # エポックごとの損失を表示
    # if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# モデルを保存
torch.save(model.state_dict(), "./models/model.pth")

print("Training Done")



##### 評価 #####
### モデル
model.load_state_dict(torch.load("./models/model.pth"))

model.eval()

with torch.no_grad():
    # 訓練データで調べる

    # 正解率の計算
    # CER(文字誤り率)
    c_sum_t = 0 #全文字数
    c_ans_sum_t = 0 #文字単位の正解数
    # WER(単語誤り率)
    w_sum_t = 0 #全文字数
    w_ans_sum_t = 0 #文字単位の正解数

    # 処理
    for dset in train_data:
        test = dset[0]
        gt = dset[1]

        output_sentence_list = []
        gt_sentence_list = []
        for (x_data, g_data) in zip(test, gt):
            x = torch.cat(x_data, dim=0).reshape(len(x_data), -1)
            # モデルに入力
            output = model(x.float())
            _, predicted = torch.max(output.data, 1)

            # 予測した英語を出力する処理　と　答えの英語を出力する処理
            output_word = ""
            gt_word = ""
            for p, g in zip(predicted.numpy(), g_data.numpy()):
                p_pred = chr(ascii_to_int(p)) # 予測文字
                g_pred = chr(g) # 答えの文字

                output_word += p_pred
                gt_word += g_pred

                # CERの計算
                c_sum_t += 1
                if(p_pred == g_pred):
                    c_ans_sum_t += 1

            output_sentence_list.append(output_word)
            gt_sentence_list.append(gt_word)

            # WERの計算
            w_sum_t += 1
            # 単語単位で一致しているかを判定
            if(output_word == gt_word):
                w_ans_sum_t += 1
            
        
        # 予測の処理を出力
        output_sentence = ' '.join(output_sentence_list)
        print("Predict:", output_sentence)

        # 答えを出力
        gt_sentence = ' '.join(gt_sentence_list)
        print("Ground Truth:", gt_sentence)

    # 文字誤り率と単語誤り率を表示
    print(" ")
    print("## CER(文字誤り率) (Training Data) ##")
    print("正解率:", round((c_ans_sum_t/c_sum_t)*100, 3), "%")
    print("CER:", round(100-(c_ans_sum_t/c_sum_t)*100, 3), "%")
    print("文字総数:", c_sum_t, "正解数", c_ans_sum_t)
    print(" ")
    print("## WER(単語誤り率) (Training Data) ##")
    print("正解率:", round((w_ans_sum_t/w_sum_t)*100, 3), "%")
    print("WER:", round(100-(w_ans_sum_t/w_sum_t)*100, 3), "%")
    print("単語総数:", w_sum_t, "正解数", w_ans_sum_t)



    # テストデータで調べる

    # 正解率の計算
    # CER(文字誤り率)
    c_sum = 0 #全文字数
    c_ans_sum = 0 #文字単位の正解数
    # WER(単語誤り率)
    w_sum = 0 #全文字数
    w_ans_sum = 0 #文字単位の正解数

    # 処理
    for dset in test_data:
        test = dset[0]
        gt = dset[1]

        output_sentence_list = []
        gt_sentence_list = []
        for (x_data, g_data) in zip(test, gt):
            x = torch.cat(x_data, dim=0).reshape(len(x_data), -1)
            # モデルに入力
            output = model(x.float())
            _, predicted = torch.max(output.data, 1)

            # 予測した英語を出力する処理　と　答えの英語を出力する処理
            output_word = ""
            gt_word = ""
            for p, g in zip(predicted.numpy(), g_data.numpy()):
                p_pred = chr(ascii_to_int(p)) # 予測文字
                g_pred = chr(g) # 答えの文字

                output_word += p_pred
                gt_word += g_pred

                # CERの計算
                c_sum += 1
                if(p_pred == g_pred):
                    c_ans_sum += 1

            output_sentence_list.append(output_word)
            gt_sentence_list.append(gt_word)

            # WERの計算
            w_sum += 1
            # 単語単位で一致しているかを判定
            if(output_word == gt_word):
                w_ans_sum += 1
            
        
        # 予測の処理を出力
        output_sentence = ' '.join(output_sentence_list)
        print("Predict:", output_sentence)

        # 答えを出力
        gt_sentence = ' '.join(gt_sentence_list)
        print("Ground Truth:", gt_sentence)

    # 文字誤り率と単語誤り率を表示
    print(" ")
    print("## CER(文字誤り率) ##")
    print("正解率:", round((c_ans_sum/c_sum)*100, 3), "%")
    print("CER:", round(100-(c_ans_sum/c_sum)*100, 3), "%")
    print("文字総数:", c_sum, "正解数", c_ans_sum)
    print(" ")
    print("## WER(単語誤り率) ##")
    print("正解率:", round((w_ans_sum/w_sum)*100, 3), "%")
    print("WER:", round(100-(w_ans_sum/w_sum)*100, 3), "%")
    print("単語総数:", w_sum, "正解数", w_ans_sum)