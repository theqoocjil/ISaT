from services.knn import KnnService
from utils.load_file import load_file

feature = ""

path_train = r''
path_test = r''

knn = KnnService(load_file(path_train), feature)
knn.model_training(range())


predict = knn.predict_data(load_file(path_test))
print(predict)