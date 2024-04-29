from abc import ABC
import sys
import builtins
import os
import traceback


from DataReader.abstract_data_reader import AbstractDataReader
from TangentS.Tuple_Extraction import latex_math_to_slt_tuples, latex_math_to_opt_tuples

sys.stdout = open("stdout.txt", "w", buffering=1)
def print(text):
    builtins.print(text)
    os.fsync(sys.stdout)


class MSEDataReader(AbstractDataReader, ABC):
    def __init__(self, collection_file_path, read_slt=True):
        self.read_slt = read_slt
        self.collection_file_path = collection_file_path
        super()

    def get_collection(self, ):
        except_count = 0
        dictionary_formula_tuples = {}
        curr_line_num = 0
        num_lines = 0
        with open(self.collection_file_path, "rb") as f:
            num_lines = sum(1 for _ in f)
            line_num_out = "Number of lines: " + str(num_lines)
            print(line_num_out)
        file = open(self.collection_file_path)
        line = file.readline().strip("\n")
        while line:
            curr_line_num += 1
            if curr_line_num % 500 == 0:
                line_num_out = "Line number: " + str(curr_line_num) + " out of " + str(num_lines) + ", " + str(((100.0 * curr_line_num)/num_lines)) + "% processed"
                print(line_num_out)
            try:
                if "$" not in line:
                    temp = ""

                    while "$" not in line:
                        temp += line
                        line = file.readline()
                    line = (temp+"\n"+line).strip("\n")
                if "USD" in line or "<p" in line or "<blockquote" in line:
                    line = file.readline().strip("\n")
                    continue
                latex_string = line.split("$")[1][1:]
                formula_id = line.split("$")[2][2:]
                # latex_out = "latex string: " + latex_string
                # formula_id_out = "formula_id string: " + formula_id
                # print(latex_out)
                # print(formula_id_out)
                # print("Parsing file...")
                if self.read_slt:
                    lst_tuples = latex_math_to_slt_tuples(latex_string)
                else:
                    lst_tuples = latex_math_to_opt_tuples(latex_string)
                dictionary_formula_tuples[formula_id] = lst_tuples
                line = file.readline().strip("\n")
            except Exception as e:
                print(str(e))
                traceback.print_exc()
                except_count += 1
                print(line)
                line = file.readline().strip("\n")
            
        print(except_count)
        return dictionary_formula_tuples

    def get_collection2(self, ):
        except_count = 0
        dictionary_formula_slt_tuple = {}
        file = open(self.collection_file_path)
        text = file.read()
        file.close()
        text_parts = text.split("$$")
        i = 0
        while i < len(text_parts):
            try:
                latex_string = text_parts[i]
                formula_id = text_parts[i+1]
                i += 3
                if self.read_slt:
                    lst_tuples = latex_math_to_slt_tuples(latex_string)
                else:
                    lst_tuples = latex_math_to_opt_tuples(latex_string)
                dictionary_formula_slt_tuple[formula_id] = lst_tuples
            except:
                except_count += 1
                print(text_parts[i])
                i += 3
        print(except_count)
        return dictionary_formula_slt_tuple
#
# def main():
#     a = MSEDataReader("../formula_latex_map.csv")
#     a.get_collection()
#     print("-------------------------------")
#     a = MSEDataReader("../formula_latex_map.csv", False)
#     a.get_collection()
#
#
# if __name__ == "__main__":
#     main()
