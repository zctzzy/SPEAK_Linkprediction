import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp

def balanced_subsample(data_, label_, subsample_size=1.0):
    class_xs = []
    min_elems = None
    labels = list(np.unique(y))
    for yi in labels:
        elems = data_[(label_ == yi)]
        class_xs.append((yi, elems))
        if min_elems is None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]
        use_elems = min_elems
        if subsample_size < 1:
            use_elems = int(min_elems*subsample_size)
    xs = []
    ys = []
    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)
        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)
        xs.append(x_)
        ys.append(y_)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    return xs, ys


def unsupervised_method(attr, label):
    fpr_, tpr_, thresholds_ = roc_curve(label, attr)
    mean_tpr_ = 0.0
    mean_fpr_ = np.linspace(0., 1., 50)
    mean_tpr_ += interp(mean_fpr_, fpr_, tpr_)
    roc_auc_attr = auc(fpr_, tpr_)
    return mean_tpr_, roc_auc_attr

def get_index(name_f, name_s):
    index = []
    for i_, a in enumerate(name_f):
        for j, b in enumerate(name_s):
            if b == a:
                index.append(i_)
    return index

def process_data(names_, _name_, balanced_X_, balanced_y_, data_, test_):
    m_tpr = 0.0
    m_fpr = np.linspace(0, 1, 50)
    classifier = RandomForestClassifier()
    index = get_index(names_, _name_)
    feature = balanced_X_[:,index]
    test_data = data_[test_][:, index]

    classifier.fit(feature, balanced_y_)
    prob = classifier.predict_proba(test_data)
    fpr, tpr, thresold = roc_curve(y[test], prob[:, 1])
    m_tpr += interp(m_fpr, fpr, tpr)
    # print m_tpr
    return m_tpr

chunksize = 10 ** 5
chunks = []
loop = True

reader = pd.read_csv('data.csv', iterator=True, header=0)
while loop:
    try:
        chunk = reader.get_chunk(chunksize)
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print "Iteration is stopped."
X = pd.DataFrame(pd.concat(chunks, ignore_index=True))
names = X.columns
########################################################
# Model index
########################################################
hplp_plus_name = ['src_deg', 'dst_deg', 'src_vol', 'dst_vol', 'maxflow', 'shortpaths', 'propflow','katz',
                    'cn', 'aa', 'jc', 'pa']

speak_plus_name = ['src_deg', 'dst_deg', 'src_vol', 'dst_vol', 'maxflow', 'shortpaths','propflow', 'katz'
                    ,'cn', 'aa', 'jc', 'pa', 'tag', 'aff', 'sp']
hplp_tpr = 0.0
speak_tpr = 0.0
mean_fpr = np.linspace(0.0, 1.0, 50)
save_result = []

y = X['label'].values

aa = X['aa'].values
aa_tpr, aa_auc = unsupervised_method(aa, y)
plt.plot(mean_fpr, aa_tpr, '-s', label='AA roc (area = %0.3f)' % aa_auc, lw=2)

pa = X['pa'].values
pa_tpr, pa_auc = unsupervised_method(pa, y)
plt.plot(mean_fpr, pa_tpr, '-o', label='PA roc (area = %0.3f)' % pa_auc, lw=2)

propflow = X['propflow'].values
propflow_tpr, propflow_auc = unsupervised_method(propflow, y)
plt.plot(mean_fpr, propflow_tpr, '-*', label='Propflow roc (area = %0.3f)' % propflow_auc, lw=2)

cv = StratifiedKFold(y, n_folds=10)

data = X.values

for i, (train, test) in enumerate(cv):
    balanced_X, balanced_y = balanced_subsample(data[train], y[train])
    print '%d' % (i+1)
    label = balanced_y
    speak_tpr += process_data(names, speak_plus_name, balanced_X, balanced_y, data, test)
    # print mean_tpr[0,:]
    hplp_tpr += process_data(names, hplp_plus_name, balanced_X, balanced_y, data, test)
    speak_tpr[0] = 0.0
    hplp_tpr[0] = 0.0

speak_tpr /= len(cv)
speak_tpr[-1] = 1.0
hplp_tpr /= len(cv)
hplp_tpr[-1] = 1.0

speak_plus_auc = auc(mean_fpr, speak_tpr)
hplp_plus_auc = auc(mean_fpr, hplp_tpr)


plt.plot(mean_fpr, hplp_tpr, 'v--', label='HPLP+ mean roc (area = %0.3f)' % hplp_plus_auc, lw=2)
plt.plot(mean_fpr, speak_tpr, 'gs--', label='SPEAK mean roc (area = %0.3f)' % speak_plus_auc, lw=2)

plt.plot([0, 1], [0, 1], '--', color=(.6, .6, .6), label='Random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
