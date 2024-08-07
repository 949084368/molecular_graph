import csv
import numpy as np
from gnn.tool import set_predict_argument, get_scaler, load_args, load_data, load_model
from gnn.train import predict
from gnn.data import MoleDataSet

def predicting(args):
    print('Load args.')
    scaler = get_scaler(args.model_path)
    print('scaler',scaler)
    train_args = load_args(args.model_path)
    
    for key,value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Load data.')
    test_data = load_data(args.predict_path,args)
    print('Load model')
    model = load_model(args.model_path,args.cuda)

    test_pred = predict(model,test_data,args.batch_size,scaler)
    assert len(test_data) == len(test_pred)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    
    print('Write result.')
    write_smile = test_data.smile()
    with open(args.result_path, 'w',newline = '') as file:
        writer = csv.writer(file)
        
        line = ['Smiles']
        line.extend(args.task_names)
        writer.writerow(line)
        
        #for i in range(fir_data_len):
        for i in range(len(test_data)):
            line = []
            line.append(write_smile[i])
            # line.extend([1 if x > 0.5 else 0 for x in test_pred[i]])
            line.extend(test_pred[i])
            writer.writerow(line)

from argparse import Namespace
if __name__=='__main__':
    args = set_predict_argument()
    # args = Namespace(predict_path='Data/ziyan_AhR.csv',
    #                  model_path='AhR32/model.pt',
    #                  result_path='ziyan_result.csv',
    #                  model_type='Vit',
    #                  batch_size=1)
    # python predict.py --predict_path Data/test.csv --model_path AhR32/model.pt --result_path result.csv --model_type Vit
    predicting(args)
    