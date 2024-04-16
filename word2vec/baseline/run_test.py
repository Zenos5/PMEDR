import pandas as pd
import os
import ast

def process_answers(answer_df):
    for index, row in samples.iterrows():
        '''save answer text portion to file'''
        text_path = 'answer/text/'
        isExist = os.path.exists(text_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(text_path)
        with open(text_path + str(row['answer_id'].values[0]) +'_'+str(row['score'])+'.txt', 'w') as f:
            f.write(row['text'].values[0])

        '''save question mathml to file'''
        math_path = 'answer/math/' + str(row['answer_id'].values[0])
        isExist = os.path.exists(math_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(math_path)
        mathml = ast.literal_eval(row['mathml'].values[0])
        print(len(mathml))
        for i, a in enumerate(mathml):
            with open(math_path+'_'+str(row['score']) + '/' + str(i) + '.xml', 'w') as f:
                f.write(a)

        '''save question htmk to file'''
        html_path = 'answer/xhtml/'
        isExist = os.path.exists(html_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(html_path)

        with open(html_path + str(row['answer_id'].values[0])+'_'+str(row['score']) + '.html', 'w') as f:
            f.write(row['html_mias'].values[0])

        '''save question latext to file'''
        latex_path = 'answer/latex/' + str(row['answer_id'].values[0])
        isExist = os.path.exists(latex_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(latex_path)
        text = row['latex'].values[0]
        for i, a in enumerate(text):
            with open(latex_path + '.csv', 'w') as f:
                temp = a + '$$' + str(row['score'])
                f.write(temp)
def process_question(samples):
    for index, row in samples.iterrows():
        '''save question text portion to file'''
        text_path = 'query/text/'
        isExist = os.path.exists(text_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(text_path)
        with open(text_path + str(row['question_id'].values[0]) + '.txt', 'w') as f:
            f.write(row['text'].values[0])

        '''save question mathml to file'''
        math_path = 'query/math/' + str(row['question_id'].values[0])
        isExist = os.path.exists(math_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(math_path)
        mathml = ast.literal_eval(row['mathml'].values[0])
        print(len(mathml))
        for i, a in enumerate(mathml):
            with open(math_path + '/' + str(i) + '.xml', 'w') as f:
                f.write(a)

        '''save question htmk to file'''
        html_path = 'query/xhtml/'
        isExist = os.path.exists(html_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(html_path)

        with open(html_path + str(row['question_id'].values[0]) + '.html', 'w') as f:
            f.write(row['html_mias'].values[0])

        '''save question latext to file'''
        latex_path = 'query/latex/' + str(row['question_id'].values[0])
        isExist = os.path.exists(latex_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(latex_path)
        text = row['latex'].values[0]
        for i, a in enumerate(text):
            with open(latex_path + '.csv', 'w') as f:
                temp = a + '$$' + str(i)
                f.write(temp)


if __name__ == "__main__":
    qdf = pd.read_csv("/Users/sophi/Project/Thesis_data/question_g_5.csv")
    adf = pd.read_csv("/Users/sophi/Project/Thesis_data/answer_g_5.csv")
    samples = qdf.sample(20)
    process_question(samples)

    answer_df = adf.loc[adf['question_id'].isin(samples['question_id'].values)]
    gb = answer_df.groupby('question_id')
    ls = [gb.get_group(x) for x in gb.groups]
    for l in ls:
        process_answers(l)