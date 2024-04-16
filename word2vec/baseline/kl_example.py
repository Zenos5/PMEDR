import main
import numpy as np

if __name__ == "__main__":
    qtf= "kl_q.txt"
    atf = ["kl_a_6.txt"]
    # atf = ["kl_a.txt"]
    # atf = ["kl_a.txt", "kl_a_1.txt", "kl_a_2.txt", "kl_a_3.txt", "kl_a_4.txt", "kl_a_5.txt", "kl_a_6.txt"]
    type = "lda"
    text_result = main.text_match(qtf=qtf, atf=atf, question_id=1, path='path', type=type)
    # print((1/20)*np.log((1/20)/((1/15)*np.log(8/5))))