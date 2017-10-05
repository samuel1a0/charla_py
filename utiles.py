import pandas as pd
import os
import itertools
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process.kernels import RBF

def cargar(fold="data/sin adulterar", one_row=False, clase=0):
    """ Devuelve los datos de la bolsa indicada en el argumento 'fold'.
        one_row: Determina si los datos de cada file deben ser devueltos
        como una sola fila, o no.
        clase: nro de clase que se asignará a los datos de la bolsa.
    """
    aux = pd.DataFrame()
    files = os.listdir(fold+"/")
    for f in files:
        dfu = pd.read_csv(fold+"/"+f, index_col=[0], names=["amp"])
        if one_row:
            dfu = dfu.T
        
        dfu["clase"] = clase
        aux = pd.concat([aux, dfu], ignore_index=True)
    return aux


def load_data(dif=False, one_row=True):
    """ Devuelve todos los datos en un solo dataframe.
        one_row: Determina si los datos de cada file deben ser devueltos
        como una sola fila, o no.
        dif: Determina si cada bolsa se cargará con un nro de clase
        diferente (True), o solo se diferenciará la clase 'sin adulterar'(False).
    """
    sin_adulterar = cargar(one_row=one_row)
    adulterados = pd.DataFrame()
    
    folds = os.listdir("data/")
    clases = pd.DataFrame(index=folds)
    folds.remove("sin adulterar")
    clases["sin adulterar"] = 0
    for i, f in enumerate(folds):
        if dif:
            ad = cargar("data/"+f, one_row=one_row, clase=i+1)
            clases[f] = i+1
        else:
            ad = cargar("data/"+f, one_row=one_row, clase=1)
            clases[f] = 1
        adulterados = pd.concat([adulterados,ad], ignore_index=True)
    data_comp = pd.concat([sin_adulterar, adulterados], ignore_index=True)
    return clases, data_comp

def load_par(folds, one_row=True):
    """ Carga los archivos de las 2 carpetas indicados en 'folds', sustrayendoles a sus
        valores el promedio de los datos 'sin adulterar' y los devuelve en un dataframe.
        one_row: Determina si los datos de cada archivo deben ser devueltos
        como una sola fila, o no.
    """
    sin = cargar(one_row=one_row)
    sin = pd.DataFrame(sin.mean(axis=0)).T

    comp = pd.DataFrame()
    resta = pd.DataFrame()
    d1 = cargar("data/"+folds[0], one_row=one_row, clase=0)
    d2 = cargar("data/"+folds[1], one_row=one_row, clase=1)
    comp = pd.concat([d1, d2], ignore_index=True)
    
    for index, row in comp.iloc[:,:-1].iterrows():
        aux = sin.iloc[:,:-1] - row
        resta = pd.concat([resta, aux], ignore_index=True)
    resta["clase"] = comp["clase"]
    return resta
    

def load_data_dif(clase="sin adulterar", dif=False, one_row=True):
    """ Devuelve todos los datos en un solo dataframe.
        one_row: Determina si los datos de cada file deben ser devueltos
        como una sola fila, o no.
        dif: Determina si cada bolsa se cargará con un nro de clase diferente
        (True), o solo se diferenciará la bolsa indicada en 'clase'(False).
    """
    clase_0 = cargar("data/"+clase, one_row=one_row)
    adulterados = pd.DataFrame()
    
    folds = os.listdir("data/")
    clases = pd.DataFrame(index=folds)
    folds.remove(clase)
    clases[clase] = 0
    for i, f in enumerate(folds):
        if dif:
            ad = cargar("data/"+f, one_row=one_row, clase=i+1)
            clases[f] = i+1
        else:
            ad = cargar("data/"+f, one_row=one_row, clase=1)
            clases[f] = 1
        adulterados = pd.concat([adulterados,ad], ignore_index=True)
    data_comp = pd.concat([clase_0, adulterados], ignore_index=True)
    return clases, data_comp


def plot_confusion_matrix(cm, classes, title='Confusion matrix', norm=True, size=(4,4)):
    """
    Esta función imprime una matriz de confución. El parámetro 'norm' determina si los valores deben
    mostrarse como recuento total de las ocurrencias, o como medida porcentual donde 1 es el total
    """
    cm_2 = cm.copy()
    if norm:
        cm = np.around(cm / cm.sum(axis=1)[:, np.newaxis], 2)
    thresh = cm.max() / 2.
    
    cmap = plt.cm.Blues
    fig = plt.figure(title, figsize=size)
    plt.clf()
    plt.title(title)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(cm), cmap=plt.cm.Blues, 
                    interpolation='nearest')
    fig.colorbar(res)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        aux = cm[i, j]
        if aux == 0:
            continue
        aux_2 = "{} ({})".format(aux, cm_2[i,j])
        plt.text(j, i, str(aux_2),
                 horizontalalignment="center",
                 color="r")

    plt.tight_layout()
    plt.xlabel('Predicted Label')
    plt.ylabel('True label')
    plt.show()
    
    
def plot_matrix_on_ax(ax, cm, classes, title='Confusion matrix', norm=True):
    """
    Imprime la matriz de confución 'cm'. El parámetro 'norm' (normalizado) determina si los valores deben
    mostrarse como recuento total de las ocurrencias, o como medida porcentual donde 1 es el total
    """
    if norm:
        cm = np.around(cm / cm.sum(axis=1)[:, np.newaxis], 2)
    thresh = cm.max() / 2.
    
    cmap = plt.cm.Blues
    ax.set_aspect(1)
    res = ax.imshow(np.array(cm), cmap=plt.cm.Blues, 
                    interpolation='nearest', vmin=0, vmax=1)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        aux = cm[i, j]
        if aux == 0:
            continue
        ax.text(j, i, str(aux), size=15,
                 horizontalalignment="center",
                 color="r")

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)#, rotation=45)
    ax.set_yticks(tick_marks, classes)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return res

def plot_data(data, clases, tit="graph"):    
    fig = plt.figure(tit)
    ax = plt.subplot(111)
    # cmap = plt.cm.jet
    cmap = plt.cm.nipy_spectral
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the .jet map
    colors = cmap.from_list('Custom cmap', cmaplist, cmap.N) # create the new map
    result = ax.scatter(data[:, 0], data[:, 1], c=clases, cmap=colors)
    fig.colorbar(result)
    plt.show()

    
def plot_data_train_test(data, clases, train, test, names=["S/A", "AD"]):
    tit="graph"
    fig = plt.figure(tit, figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1)
    cmap = plt.cm.nipy_spectral
    # cmap = plt.cm.winter
    cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the .jet map
    colors = cmap.from_list('Custom cmap', cmaplist, cmap.N) # create the new map
    if len(np.unique(clases)) == len(names):
        # l = [names[x] for x in np.unique(clases)]
        plt.scatter(data[:, 0], data[:, 1], c=clases, cmap=colors)
        # plt.legend(l, loc=3)
    else:
        plt.scatter(data[:, 0], data[:, 1], c=clases, cmap=colors)

    plt.title("Data")
    
    plt.subplot(1, 3, 2, sharey=ax1, sharex=ax1)
    plt.scatter(train["x"][:, 0], train["x"][:, 1], c=train["y"], cmap=colors)
    plt.title("Train")
    
    plt.subplot(1, 3, 3, sharey=ax1, sharex=ax1)
    plt.scatter(test["x"][:, 0], test["x"][:, 1], c=test["y"], cmap=colors)
    plt.title("Test")

    cb = plt.colorbar()
    tope = max(np.unique(clases))+1
    loc = np.arange(0, tope, max(np.unique(clases))/(len(np.unique(clases))-1) )
    cb.set_ticks(loc)
    cb.set_ticklabels(names)
    plt.show()


    
def clasificar(train_x, train_y, test_x, test_y, clases, kernel="linear", grado=2):
    a_dtree = tree.DecisionTreeClassifier(criterion="entropy", max_features="log2", random_state=15, max_depth=10)
    a_etree = ExtraTreesClassifier(n_jobs=4, random_state=15)
    a_mlp = MLPClassifier(alpha=1, random_state=15)
    # Gaussian con 'multi_class=one_vs_one' clasifica sin error como los otros
    a_gpc = GaussianProcessClassifier(n_jobs=4, multi_class="one_vs_rest", random_state=15)
    a_knn = KNeighborsClassifier(n_jobs=4)
    a_svc = SVC(C=1, kernel=kernel)

    cls = [("DecisionTreeClassifier", a_dtree),
           ("ExtraTreesClassifier", a_etree),
           ("MLPClassifier", a_mlp),
           ("GaussianProcessClassifier", a_gpc),
           ("KNeighborsClassifier", a_knn),
           ("SVC '{}'(degree {})".format(kernel, grado), a_svc)]

    coord = [(x,y) for x, y in itertools.product(range(2), range(3))]
    fig, axarr = plt.subplots(2, 3, figsize=(12, 7))

    plt.xticks(rotation=70)
    
    for (x, y), (n, c) in zip(coord, cls):
        c.fit(train_x, train_y)

        a_predict = c.predict(test_x)
        a_mat = confusion_matrix(test_y, a_predict)
        im = plot_matrix_on_ax(axarr[x, y], a_mat, clases, title=n)
        plt.tight_layout()
        
    bar = fig.add_axes([1.05, 0, 0.03, 1])
    fig.colorbar(im, cax=bar)
    plt.show()
    return


class DataHandler:
    class __DataHandler:
        path = None
        datos = {}
        
        def __init__(self, path):
            self.path = path
        def __str__(self):
            return repr(self) + self.path

        def get_data(self, una_bolsa, one_row=True):
            if una_bolsa in self.datos.keys():
                return self.return_data(una_bolsa, one_row)

            return self.load_data(una_bolsa, one_row)
        
        def return_data(self, una_bolsa, one_row=False):
            bolsa = pd.DataFrame([])
            keys = self.datos[una_bolsa].keys()
            for k in keys:
                df = self.datos[una_bolsa][k]
                bolsa = pd.concat([bolsa, df], ignore_index=True)
            return bolsa
        
        
        def load_data(self, una_bolsa, one_row=False):
            sets = os.listdir(self.path+"/"+una_bolsa)
            if len(sets) != 0:
                self.datos[una_bolsa] = {}
                for s in sets:
                    ruta = "{}/{}/{}".format( self.path, una_bolsa, s)
                    df = pd.read_csv(ruta, index_col=[0], names=["amp"]).T
                    self.datos[una_bolsa][s] = df
                return self.return_data(una_bolsa, one_row)
            print("Error!: No existe la bolsa especificada")
            return
        
        def get_all(self, one_row=True):
            total_data = pd.DataFrame()
            for i, k in enumerate(self.datos.keys()):
                df = self.return_data(k, one_row)
                df["clase"] = k
                total_data = pd.concat([total_data, df], ignore_index=True)
            return total_data
        
        def load_all(self):
            if self.path:
                bolsas = [x.name for x in os.scandir(self.path) if os.path.isdir(x)]
                for b in bolsas:
                    self.load_data(b)
            else:
                print("Path undefined")

        
    instance = None
    def __init__(self, path):
        if not DataHandler.instance:
            DataHandler.instance = DataHandler.__DataHandler(path)
        else:
            DataHandler.instance.path = path
    def __getattr__(self, name):
        return getattr(self.instance, name)