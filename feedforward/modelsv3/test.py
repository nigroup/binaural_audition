import pdb

import numpy as np


content = []
content.append(("nr_conv_layers_ratemap", 1))
content.append(("nr_conv_layers_ams", 2))
content.append(("number_fully_connected_layers",3))

dict = {}

for listelement in content:
    dict[listelement[0]] = listelement[1]

print(content)