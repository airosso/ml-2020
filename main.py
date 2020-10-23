import itertools

import dateutil
import numpy as np
import pandas
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, FeatureUnion
from xgboost import XGBClassifier

from preprocessing import text_preprocessing, liwc


def cross_validate(x, y, modelFn, params):
    results = []
    for slice_train, slice_test in KFold(n_splits=3).split(x):
        x1 = x[slice_train]
        y1 = y[slice_train]
        x2 = x[slice_test]
        y2 = y[slice_test]
        model = modelFn(**params)
        model.fit(x1, y1)
        y3 = model.predict(x2)
        results.append(f1_score(y2, y3, average='macro'))
    i = params.copy()
    i['score'] = sum(results) / len(results)
    return i


def best(results):
    scoreMax = None
    p = None
    for r in results:
        score = r['score']
        if scoreMax is None or score > scoreMax:
            scoreMax = score
            p = r.copy()
            p.pop('score')
    return p


def parameters(**params):
    return map(lambda m: dict(m), itertools.product(*[[(k, v1) for v1 in v] for k, v in params.items()]))


def train_organization_classification(train, test):
    train = train.sample(frac=1)
    x1, y1 = train['TweetText'].to_numpy(), train['Topic'].to_numpy()
    x2, y2 = test['TweetText'].to_numpy(), test['Topic'].to_numpy()

    def modelFn(max_features):
        return make_pipeline(
            TfidfVectorizer(max_features=max_features),
            svm.SVC(probability=True)
        )

    results = []
    for params in parameters(max_features=[100, 200]):
        results.append(cross_validate(x1, y1, modelFn, params))
    print(results)
    b = modelFn(**best(results)).fit(x1, y1)
    print(f1_score(y2, b.predict(x2), average='macro'))
    return b


def train_sentiment_classification(train, test, organization_estimator):
    train = train.sample(frac=1)

    def pack_date(ds):
        text = ds['TweetText'].to_numpy()
        date = ds['TweetDate'].to_numpy()
        return np.array(list(zip(text, date)))

    x1, y1 = pack_date(train), train['Sentiment'].to_numpy()
    x2, y2 = pack_date(test), test['Sentiment'].to_numpy()

    def modelFn(max_features, enableAdditionalFeatures, n_estimators):
        return make_pipeline(
            FeatureUnion([
                ('a', TfIdfWrapper(TfidfVectorizer(max_features=max_features))),
                ('b', AdditionalFeatures(
                    enableAdditionalFeatures,
                    [
                        DateTransformer(),
                        OrgTransformer(organization_estimator),
                        DictionaryTransformer()
                    ]
                ))
            ]),
            XGBClassifier(n_estimators=n_estimators)
        )

    results = []
    for params in parameters(
            max_features=[1000, 1500, 2000],
            n_estimators=[100, 600, 1000],
            enableAdditionalFeatures=[True, False],
    ):
        result = cross_validate(x1, y1, modelFn, params)
        b = modelFn(**params).fit(x1, y1)
        print(params)
        print(f1_score(y2, b.predict(x2), average='macro'))
        results.append(result)
    print(results)
    b = modelFn(**best(results)).fit(x1, y1)
    print(f1_score(y2, b.predict(x2), average='macro'))
    return b


class DateTransformer:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x, y, **kwargs):
        self.min = np.min([self.parse(i[1]) for i in x])
        self.max = np.max([self.parse(i[1]) for i in x])

    def parse(self, x):
        return dateutil.parser.parse(x).timestamp()

    def transform(self, x):
        i = [self.parse(i[1]) for i in x]
        return np.divide(np.subtract(i, self.min), self.max - self.min)[np.newaxis].T


class DictionaryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y):
        return self

    def transform(self, x):
        return [[liwc(i[1])[4]] for i in x]


class OrgTransformer:
    def __init__(self, f):
        self.f = f

    def fit(self, x, y, **kwargs):
        return self

    def transform(self, x):
        return self.f.predict_proba([i[0] for i in x])


class AdditionalFeatures:
    def __init__(self, enable, transformers):
        self.enable = enable
        self.transformers = transformers

    def transform(self, x):
        if self.enable:
            return np.concatenate([transformer.transform(x) for transformer in self.transformers], axis=1)
        else:
            return [[0.0] for i in x]

    def fit(self, x, y, **kwargs):
        for i in self.transformers:
            i.fit(x, y)
        return self


class TfIdfWrapper:
    def __init__(self, tf):
        self.tf = tf

    def fit(self, x, y, **kwargs):
        self.tf = self.tf.fit([i[0] for i in x], y, **kwargs)
        return self

    def transform(self, x):
        result = self.tf.transform([i[0] for i in x])
        return result


def main():
    def read(file):
        p = pandas.read_csv(file)
        p['TweetText'] = p['TweetText'].apply(lambda text: text_preprocessing(text))
        p = p[p['Sentiment'] != 'irrelevant']
        p = p[p['TweetText'].apply(lambda text: len(text) > 10)]
        p['Sentiment'] = p['Sentiment'].apply(lambda s: 1 if s == 'positive' else 0 if s == 'neutral' else -1)
        return p

    train, test = read("Train.csv").sample(frac=1), read("Test.csv")

    organization_predictor = train_organization_classification(train, test)
    sentiment_predictor = train_sentiment_classification(train, test, organization_predictor)


if __name__ == '__main__':
    main()
