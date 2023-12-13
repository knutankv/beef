import dill

def save(analysis, path):
    with open(path, 'wb') as f:
        dill.dump(analysis, f, -1)
def load(path):
    with open(path, 'rb') as f:
        model = dill.load(f)
    return model