import shutil
import sys
import os

import sentencepiece as spm

if __name__ == "__main__":
    max_turns = [1, 2]
    data_folder = "data/sach_mem/train"

    task = sys.argv[1]

    for max_turn in max_turns:
        turn_folder = f"{data_folder}/max_turn_{max_turn}"

        # train sentencepiece for each max_turn
        if task == "train":
            model_file = f"{turn_folder}/spm.model"
            vocab_file = f"{turn_folder}/spm.vocab"

            spm.SentencePieceTrainer.Train(
                f' --input={turn_folder}/texts.txt'
                f' --model_prefix=spm'
                f' --vocab_size=10000'
                f' --model_type=word'
                f' --character_coverage=1.0'
                f' --hard_vocab_limit=false'
                # f' --user_defined_symbols=<custom>'
                f' --unk_id=3'
                f' --bos_id=1'
                f' --eos_id=2'
                f' --pad_id=0'
            )

            shutil.move("spm.model", model_file)
            shutil.move("spm.vocab", vocab_file)

        # test some texts
        elif task == "test":
            sp = spm.SentencePieceProcessor()
            sp.load(f"{turn_folder}/spm.model")

            test_texts = [
                "chào cô có thể lập cho cháu lick đc ko ạ",
                "minh muon tao tai khoan __eou__ cho sach mem",
                "chào sfs sg sf cô <pad> <pad>", # special tokens must be handled outside sentencepiece
                "<s> cô </s>" # special tokens must be handled outside sentencepiece
            ]
            for test_text in test_texts:
                ids = sp.EncodeAsIds(test_text)
                decoded_text = sp.DecodeIds(ids)

                print (f"Test max_turn {max_turn} model")
                print ("text: ", test_text)
                print ("ids: ", ids)
                print ("decode: ", decoded_text)
                print ("test_text == decoded_text: ", test_text == decoded_text)
