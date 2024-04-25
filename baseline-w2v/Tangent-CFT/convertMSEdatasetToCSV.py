import os
import glob
import pathlib
import csv
from lxml import etree

def mathml2latex_yarosh(equation):
    """ MathML to LaTeX conversion with XSLT from Vasil Yaroshevich """
    xslt_file = os.path.join('mathconverter', 'xsl_yarosh', 'mmltex.xsl')
    dom = etree.fromstring(equation)
    xslt = etree.parse(xslt_file)
    transform = etree.XSLT(xslt)
    newdom = transform(dom)
    return str(newdom)

def make_corpus_csv(data_path):
    with open('MSE_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["latex", "formula_id"]
        writer.writerow(field)
        qa_list = glob.glob(os.path.join(data_path, '*'))
        for qa in qa_list:
            qa_type = "answers"
            file_dir = os.path.join(qa, qa_type)
            file_list = os.listdir(file_dir)
            if len(file_list) <= 0:
                print(file_dir, " is empty")
            for file_name in file_list:
                # print("file_name:",file_name)
                tag = pathlib.Path(qa).stem + "_" + file_name[0:file_name.index(".")]
                file_path = os.path.join(file_dir, file_name)
                with open(file_path, 'r') as f:
                    text = f.read()
                    latex = mathml2latex_yarosh(text)
                    writer.writerow([latex, tag])


if __name__ == '__main__':
    data_dir = "../../data/MSE_dataset_full/dataset_full/math/"
    make_corpus_csv(data_dir)