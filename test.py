from model.classifier import *
import pandas as pd

if __name__ == '__main__':
    dataset = 'Amazon'

    classifier = TextCNN()
    classifier.load_model('model/' + dataset + '/TextCNN/model_victim')

    # df_test = pd.read_csv('corpus/'+dataset+'/test.csv')
    # y_predict = classifier.predict_classes(list(df_test['review'])[0:500])
    # y_reality = df_test['label'].tolist()[0:500]
    # count = 0
    # for i in range(0, len(y_predict)):
    #     if y_predict[i] == y_reality[i]:
    #         count += 1
    # print(count / len(y_predict))

    with open('corpus/Amazon/irony_TextCNN.txt') as f:
        y_predict = classifier.predict_classes(f.readlines())
        y_reality = [0 for i in range(0, 500)]
    count = 0
    for i in range(0, len(y_predict)):
        if y_predict[i] == int(y_reality[i]):
            count += 1
    print(count / len(y_predict))
