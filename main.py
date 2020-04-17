import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import argparse
import json

from train_agent import TrainAgent
from test_agent import TestAgent
from prepare_data import load_data


def parse_args():
    '''
        parsing and configuration
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str,
                        default="standard",
                        help="[%(default)s] A string to describe this model")
    parser.add_argument("--data", type=str,
                        default='pmnist',
                        choices=['pmnist','mnist','add','copy'],
                        help="[%(default)s] Path to the dataset.")
    parser.add_argument("--layers", type=str,
                        default="128",
                        help="[%(default)s] A comma-separated list"
                        " of the layer sizes")
    parser.add_argument("--stack", type=int,
                        default=4,
                        help="[%(default)s] The batch size to train with")
    parser.add_argument("--batch_size", type=int,
                        default=200,
                        help="[%(default)s] The batch size to train with")
    parser.add_argument("--keep_prob", type=float,
                        default=0.9,
                        help='[%(default)s] The keep probability to use'
                        ' for training')
    parser.add_argument('--max_grad_norm', type=float,
                        default=5.0,
                        help='[%(default)s] The maximum grad norm to clip by')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001,
                        help='[%(default)s] The learning rate to train with')
    parser.add_argument('--optimizer', type=str,
                        default='adam',
                        choices=['momentum', 'rms', 'adam'],
                        help='[%(default)s] The optimizer to train with')
    parser.add_argument("--epochs", type=int,
                        default=100,
                        help="[%(default)s] The number of epochs to train for")
    parser.add_argument("--test", action='store_true',
                        help="[False] If True, the model "
                        "is only tested and not trained.")
    parser.add_argument("--logdir", type=str,
                        default="log",
                        help="[%(default)s] The directory to write"
                        " tensoboard logs to")
    parser.add_argument("--gpu", type=str,
                        default=None,
                        help="[%(default)s] The specific GPU to train on.")
    parser.add_argument('--wd', type=float,
                        default=0.0,
                        help='[%(default)s] weight decay importance')
    parser.add_argument('--results_file', type=str,
                        default=None,
                        help='[%(default)s] The file to append results to. '
                        ' If set, nothing else will be logged or saved.')
    parser.add_argument('--chrono', type=bool, default=True,
                        help='[False] If set, chrono-initialization is used.')
    parser.add_argument('--log_test', action='store_true',
                        help='[False] Log test data metrics on TB.')
    parser.add_argument('--cell', type=str,
                        default='bn-star',
                        choices=['rnn','lstm','star','bn-star'],
                        help='[%(default)s] The type of cell to use.')
    parser.add_argument("--T", type=int,
                        default=200,
                        help="[%(default)s] Sequence length for add/copy.")
    parser.add_argument("--log_every", type=int,
                        default=200000,
                        help="[%(default)s] How often to log highres loss.")

    return parser.parse_args()


def test_wrapper(test_agent, args):
    print('test score')
    data_list = load_data(args.data)
    x_test = data_list[4]
    y_test = data_list[5]
    test_agent.test(x_test, y_test, 'models/'+args.name+'/')

def test_valid_wrapper(test_agent, args):
    print('validation score')
    data_list = load_data(args.data)
    x_test = data_list[2]
    y_test = data_list[3]
    test_agent.test(x_test, y_test, 'models/'+args.name+'/')
    
def main(args):
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        try:
            import py3nvml
            py3nvml.grab_gpus(1, gpu_fraction=0.95)
        except ImportError:
            print("Could not import py3nvml")

    if args.test:
        # Get the config
        with open(os.path.join('models',args.name,'config.json')) as fp:
            config_dict = json.load(fp)
        args_dict = vars(args)
        args_dict.update(config_dict)

        test_agent = TestAgent(args)
        test_wrapper(test_agent, args)
        valid_agent = TestAgent(args)
        test_valid_wrapper(valid_agent, args)


    else:
        agent = TrainAgent(args)
        test_agent = TestAgent(args)
        valid_agent = TestAgent(args)
        try:
            agent.train(args.data, args.max_grad_norm, args.wd,
                        test_agent, args=args)
            test_valid_wrapper(valid_agent, args)
            #test_wrapper(test_agent, args)
        except KeyboardInterrupt:
            test_valid_wrapper(valid_agent, args)
            test_wrapper(test_agent, args)


if __name__ == "__main__":
    main(args=parse_args())
