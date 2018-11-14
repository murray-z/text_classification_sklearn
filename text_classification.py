# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
import jieba
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
tfidf_model_path = os.path.join(HERE, 'model', 'tfidf.pkl')
label_idx_path = os.path.join(HERE, 'map', 'label_to_idx.json')
idx_label_path = os.path.join(HERE, 'map', 'idx_to_label.json')
stopwords_path = os.path.join(HERE, 'data', 'stop_words.txt')


class TextClassification():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)

    def load_stopwords(self, stopwords_path=None):
        if stopwords_path:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        return []

    def preprocess_data(self, corpus_path):
        """
        数据预处理
        :param corpus_path: 语料路径，每行一条文本，“标签\t文本”
        :param map_path: 语料路径，每行一条文本，“标签\t文本”
        :return:
        """
        corpus = []
        labels = []

        label_to_idx = {}
        idx_to_label = {}

        idx = 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[:1000]:
                lis = line.strip().split('\t')
                corpus.append(' '.join([word for word in jieba.lcut(lis[1]) if word not in self.stopwords]))
                labels.append(lis[0])
                if lis[0] not in label_to_idx:
                    label_to_idx[lis[0]] = idx
                    idx_to_label[idx] = lis[0]
                    idx += 1

        y = [label_to_idx[label] for label in labels]

        self.dump_json(label_to_idx, label_idx_path)
        self.dump_json(idx_to_label, idx_label_path)

        return corpus, y

    def dump_json(self, obj, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)

    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.loads(f.read(), encoding='utf-8')

    def save_model(self, obj, model_path):
        joblib.dump(obj, model_path)

    def load_model(self, model_path):
        return joblib.load(model_path)

    def train(self, corpus_path, model='svm'):
        print('preprocess data ......')
        corpus, y = self.preprocess_data(corpus_path)

        print('transform data to tfidf ......')
        tfidfvertorizer = TfidfVectorizer()
        X = tfidfvertorizer.fit_transform(corpus)

        print('save tfidf model .....')
        self.save_model(tfidfvertorizer, tfidf_model_path)

        print('split data ......')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

        MODEL = None
        if model == 'knn':
            MODEL = KNeighborsClassifier()

        if model == 'svm':
            MODEL = SVC()

        if model == 'random_forest':
            MODEL = RandomForestClassifier()

        print('train model ......')
        MODEL.fit(train_x, train_y)

        print('evaluate model .....')
        y_pred = MODEL.predict(test_x)
        print(metrics.classification_report(test_y, y_pred))

        model_path = os.path.join(HERE, 'model', '{}.pkl'.format(model))

        print('save model .....')
        self.save_model(MODEL, model_path)

    def predict(self, text, model):
        """
        预测
        :param text: 文本
        :param model: 模型 [knn/svm/random_forest]
        :return:
        """
        model_path = os.path.join(HERE, 'model', '{}.pkl'.format(model))

        tfidf_model = self.load_model(tfidf_model_path)

        MODEL = self.load_model(model_path)

        idx_to_label = self.load_json(idx_label_path)

        x = tfidf_model.transform([' '.join([word for word in jieba.lcut(text) if word not in self.stopwords])])

        category = MODEL.predict(x)

        print(category)

        print('label: {}'.format(idx_to_label[str(category[0])]))


if __name__ == '__main__':
    text = """纽约操盘手暗示小斯令人失望他离我们的期待很远新浪体育讯北京时间3月29日，作为号称全球第一都市的纽约，尼克斯的一举一动自然会成为媒体和球迷的关注焦点。虽然在交易截止日前弄来卡梅罗-安东尼，让整个联盟为之一振。不过，据《纽约邮报》消息，尼克斯的操盘手唐尼-沃尔什，对此并不满意。“我们还有很多工作要做，过去有很多工作要做，现在还是一样，”沃尔什说，“交易前，我们缺少必要的补强，现在，问题依旧没有得到解决。”难道，是沃尔什对得到安东尼并不满意？所幸，沃尔什对此做出了解释。在他看来，球队现在的问题并不是安东尼，而是阿玛雷-斯塔德迈尔，他在内线的孤立无援才是问题根本。“我看到的是，我还得弄来球员去帮助阿玛雷。”沃尔什解释说。“他能把本职工作做到最佳，不过这距离我们对他的期望还是有差距的。我知道我该怎么做才能帮助球队，我们还需要内线的大个子，越多越好。”的确，对尼克斯来说，斯塔德迈尔的技术特点非常鲜明，他是一个得分能力不错的强力大前锋，但是以他的体格来说，篮板能力和内线防守却实在难以匹配。特别是在和魔术的对比中，德怀特-霍华德更是让纽约人倍感郁闷。在内线羸弱的情况下，尼克斯目前还是保持着东部第七的排名，进入季后赛问题应该不大。沃尔什还为此特别夸赞了主帅迈克-德安东尼，“他对球队的问题是没什么责任的，因为短时间内他也做不了更多。所以，这一切应该都是我的责任，”沃尔什力挺德安东尼说，“所以，我通常不会在赛季中做交易，这对主帅是个太大的挑战。今年做了，是因为从长远看，对球队还是有好处的。”除了分清责任之外，沃尔什也坦诚眼下的尼克斯还是比较让他失望的，“尽管这是一笔着眼未来的交易，但还是让球队无可避免地陷入分裂，而且很难找回过去的默契程度。我很喜欢球队交易之前的状态，我不认为我们能赢得总冠军。我们打的并不好，作为整体，我们表现的很纠结。在其他球队拼命为季后赛或者更好的目标冲刺的时候，我们在攻防两端都出现了问题。”听的出来，沃尔什对尼克斯的现状是非常不满，而且几乎是在放话说球队还要有交易。季后赛之前，搞出这样的风波，不知道安东尼和斯塔德迈尔该做何感想了。(XWT185)"""
    text_cls = TextClassification(stopwords_path)
    text_cls.train('./data/test_data.txt', model='knn')
    text_cls.predict(text, model='knn')
