from SceneInstance import SceneInstance
from Scene import *
from Group import *





s = Scene("train", 1)
for i in np.arange(labelcats.shape[0]):
    s.plotSingleLabelLengthDistribution(i)
