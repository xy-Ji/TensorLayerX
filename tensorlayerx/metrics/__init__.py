#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorlayerx.backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_metric import *
elif BACKEND == 'mindspore':
    from .mindspore_metric import *
elif BACKEND == 'paddle':
    from .paddle_metric import *
elif BACKEND == 'torch':
    from .torch_metric import *
elif BACKEND == 'oneflow':
    from .oneflow_metric import *
else:
    raise NotImplementedError("This backend is not supported")
