from SELFRec import SELFRec
from util.conf import ModelConf
import warnings
warnings.filterwarnings("ignore", category=Warning)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    # Register your model here
    graph_baselines = ['LightGCN','DirectAU','MF','SASRec']
    ssl_graph_models = ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'MHCN_RETWEET', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL','MixGCF']
    sequential_baselines= ['SASRec']
    ssl_sequential_models = ['CL4SRec']

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

    print('Graph-Based Baseline Models:')
    print('   '.join(graph_baselines))
    print('-' * 100)
    print('Self-Supervised  Graph-Based Models:')
    print('   '.join(ssl_graph_models))
    print('=' * 80)
    print('Sequential Baseline Models:')
    print('   '.join(sequential_baselines))
    print('-' * 100)
    print('Self-Supervised Sequential Models:')
    print('   '.join(ssl_sequential_models))
    print('=' * 80)
    # model = input('Please enter the model you want to run:')
    # model = 'MHCN_RETWEET'
    model = 'MHCN'
    print(f'Selecting {model}')
    import time

    s = time.time()
    if model in graph_baselines or model in ssl_graph_models or model in sequential_baselines or model in ssl_sequential_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f mins" % ((e - s)/60))
