import os
from prettytable import PrettyTable

PATH_TEST = "tests" #personality tests root folder
PATH_IMG= "images" #images root folder


class Data():
    """
    Data is a type that reference a handwriting image and the associated test
    """
    def __init__(self, img_path, test_path) -> None:
        self.img_path = img_path
        self.test_path = test_path

class Dataset():
    """
    Dataset is a type that contains a list of Data types
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def add_data(self, data: Data):
        self.data_list.append(data)

    def delete_data(self, index):
        del self.data_list[index]

    def filter_data(self, filter_func):
        return list(filter(filter_func, self.data_list))

    def show(self):
        tabla = PrettyTable()
        tabla.field_names = ['Index', 'ID', 'Images', 'Tests']

        index = 0
        for d in self.data_list:
            tabla.add_row([index , index + 1, d.img_path, d.test_path])
            index += 1

        print(tabla)

def generate_path(folder_path):
    """
    params: folder_path -> The inicial folder for start listing files
    outputs: files_path -> A list with all files, alphabetically sorted by both folder and file
    """
    files_path = []
 
    # Get the path of the current working directory + tests root folder
    cwd = os.getcwd() + "/" + folder_path

    for dirpath, dirnames, filenames in os.walk(cwd):
        for name in sorted(dirnames + filenames):
            full_path = os.path.join(dirpath, name)
            if os.path.isfile(full_path):
                files_path.append(full_path)

    return files_path
          
def main():
    # Generate list of img and test files
    tests = generate_path(PATH_TEST)
    images = generate_path(PATH_IMG)

    # List to save the Data types
    data = []

    # Generate Data types with paths
    for i in range(len(tests)):
        data.append(Data(images[i], tests[i]))

    # Create Dataset
    dataset = Dataset(data)

    # Print Dataset
    dataset.show()

if __name__ == '__main__':
    main()



