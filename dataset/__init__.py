def create_dataset(name, level=1):
    module = __import__(name, globals(), locals(), level=level)
    return module.Dataset