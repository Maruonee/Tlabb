from tensorflow.compat.v1 import ConfigProto, InteractiveSession

#GPU fix
def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print(session)
fix_gpu()