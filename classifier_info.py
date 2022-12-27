
class ClassifierInfo:
    def __init__(self, species:str, epoch:float):
        self.species = species
        self.epoch = epoch

classifiers_info = {
    "hyena": ClassifierInfo("hyena", 3.0),
    "meerkat": ClassifierInfo("meerkat", 2.0),
    "coati": ClassifierInfo("coati", 1.0)
}
