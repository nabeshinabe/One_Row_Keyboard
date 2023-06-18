import os
import glob
import re

## ConvertDataに置いてあるファイルをルールに従って数字列に変換し、OneRowDataに格納 ##
documents = []

# Dataディレクトリ内全ての文書を獲得
path_data = "./RowData"
files = glob.glob(f"{path_data}/*")
print(files)


path_OneRowData = "./OneRowData"
path_Convertdata = "./ConvertData"

for file in files:
    with open(file, 'r') as f:
        print(os.path.basename(file))
        # documents.append([os.path.basename(file), f.read()])
        sentence = f.read()

        sentence_sent = ""
        onerow_sent = ""

        for c in sentence:
            asnii_c = ord(c) 

            if asnii_c < 32: # LF
                continue
            elif asnii_c == 32: # SPC
                sentence_sent += " "
                onerow_sent += " "
            else:
                sentence_sent += c

                asnii_c %= 10
                onerow_sent += str(asnii_c)

        sentence_sent = re.sub(r"\s+", " ", sentence_sent)
        onerow_sent = re.sub(r"\s+", " ", onerow_sent)

        print(sentence_sent)
        print(onerow_sent)

        with open(f"{path_Convertdata}/{os.path.basename(file)}_base.txt", mode='w') as f:
            f.write(sentence_sent)
        
        with open(f"{path_OneRowData}/{os.path.basename(file)}_onerow.txt", mode='w') as f:
            f.write(onerow_sent)

print(len(files))         
        

