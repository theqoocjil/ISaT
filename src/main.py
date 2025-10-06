from services.knn import KnnService
from utils.load_file import load_file

feature = "phone_type"

path_train = r'/Users/theqoocjil/Documents/DataSetCSV.csv'
path_test = r'/Users/theqoocjil/Documents/DataSetTestCSV.csv'

knn = KnnService(load_file(path_train), feature)
knn.model_training(range(3, 21))


predict = knn.predict_data(load_file(path_test))
print(predict)