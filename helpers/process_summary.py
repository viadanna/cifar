from __future__ import print_function
from models import simple, simple_reg, deep, lenet, lenet_reg, deeper, mininception

model_map = {
    'simple': simple,
    'simple_reg': simple_reg,
    'deep': deep,
    'lenet': lenet,
    'lenet_reg': lenet_reg,
    'deeper': deeper,
    'mininception': mininception,
}

for name, model in model_map.items():
    print('{} model description:'.format(name))
    print(model().summary())
