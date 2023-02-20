from data.loader import FileIO

class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])
        self.dev_data = FileIO.load_data_set(config['dev.set'], config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.' + self.config['model.type'] + '.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)  # gqy: exec 执行储存在字符串或文件中的 Python 语句，相比于 eval，exec可以执行更复杂的 Python 代码
        recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,self.dev_data,**self.kwargs)'
        eval(recommender).execute()  # gqy:eval() 函数用来执行一个字符串表达式，并返回表达式的值。
