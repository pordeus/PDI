"""
Created on Thu Jun  3 12:29:07 2021

@author: daniel pordeus
Trabalho Final
Extração de features do pulmão para entrada em redes reurais
"""

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_precision_recall_curve
from skimage.feature import local_binary_pattern
from sklearn.model_selection import learning_curve
import skimage.data
import skimage.transform
import os
import glob
import os.path
from sklearn.metrics import accuracy_score#, roc_auc_score
from sklearn.model_selection import KFold

# Plotagem

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

## Fim plotagem


## FUNCOES
def novaImagem(imagem):
    imgSaida = imagem[250:750,100:900]
    return imgSaida

def binarizacao(imagem, threshold):
    img_bin = np.copy(imagem)
    for x in range(imagem.shape[0]):
        for y in range(imagem.shape[1]):
            if imagem[x,y] < threshold:
                img_bin[x,y] = 0
            else:
                img_bin[x,y] = 255
    return img_bin

def listaPixelsAcimaThreshold(threshold, imagem):
    listaPixels = []
    for x in range(imagem.shape[0]):
        for y in range(imagem.shape[1]):
            if imagem[x,y] > threshold:
                listaPixels.append([x,y])
    return np.array(listaPixels)


def pertence(G, u, v):
    for i in range(0,len(G)):
        if (G[i][0] == u):
            if(G[i][1] == v):
                return True
                #print(f"Posição {i}")
    return False

#imagem binária como entrada
def contornoImagem(imagem_binaria):
    contorno = imagem_binaria.copy()
    for x in range(0,contorno.shape[0]-1):
        first = False
        for y in range(int(np.floor(contorno.shape[1]/2))):
            if first == False and contorno[x,y] == 255:
                first = True
                last = y+1
            elif first and contorno[x,y] == 255:
                contorno[x, last] = 0
                last = y
        first = False
        for y in range(int(np.floor(contorno.shape[1]/2)), contorno.shape[1]):
            if first == False and contorno[x,y] == 255:
                first = True
                last = y+1
            elif first and contorno[x,y] == 255:
                contorno[x, last] = 0
                last = y
    return contorno

def contornoPreenchimento(imagem_binaria, original):
    contorno = imagem_binaria.copy()
    for x in range(0,contorno.shape[0]-1):
        first = False
        for y in range(int(np.floor(contorno.shape[1]/2))):
            if first == False and contorno[x,y] == 255:
                first = True
                last = y+1
            elif first and contorno[x,y] == 255:
                contorno[x, last] = original[x, last]
                last = y
        first = False
        for y in range(int(np.floor(contorno.shape[1]/2)), contorno.shape[1]):
            if first == False and contorno[x,y] == 255:
                first = True
                last = y+1
            elif first and contorno[x,y] == 255:
                contorno[x, last] = original[x, last]
                last = y
    return contorno

#Negativo de imagem cinza
def negativoImagem(imagem):
    negativo = imagem.copy()
    L = 256
    for M in range(imagem.shape[0]):
        for N in range(imagem.shape[1]):
            negativo[M,N] = L - 1 - int(imagem[M,N])
    return negativo

#Normalizacao [0,1] para vetores unidimensionais
def normalizacao01_Y(dados):
    dados_norm = np.zeros(dados.shape)
    zero = np.min(dados)
    um = np.max(dados)
    for x in range(dados.shape[0]):
        dados_norm[x] = (dados[x] - zero) / (um - zero)
    return dados_norm

#Normalizacao [0,1] para matriz [n x 2]
def normalizacao01_X(dados):
    dados_norm = np.zeros(dados.shape)
    for x in range(dados.shape[0]):
        zero = np.min(dados[x])
        um = np.max(dados[x])
        for y in range(dados.shape[1]):
            dados_norm[x,y] = (dados[x,y] - zero) / (um - zero)
    return dados_norm

#Transformada Logaritma
def transfLogaritma(imagem):
    tLog = imagem.copy()
    c = 255 / np.log(256) # L_max / log (1 + L_max)
    for M in range(0, imagem.shape[0]):
        for N in range(0, imagem.shape[1]):
            tLog[M,N] = np.round(c * np.log(int(imagem[M,N]+1)))
    return tLog

#imagem LBP como entrada
def histograma(imagem_lbp): 
    hogImage = []
    for i in range(int(np.max(imagem_lbp))+1):
        conte = np.count_nonzero(imagem_lbp == i)
        hogImage.append(conte)
    return hogImage

def processamento(imagem):
    imgSaida = np.zeros(imagem.shape)
    imgSaida = novaImagem(imagem)
    imgSaida = binarizacao(imgSaida, np.mean(imgSaida))
    imgSaida = negativoImagem(imgSaida)
    imgSaida = contornoPreenchimento(imgSaida, imagem)
    imgSaida = local_binary_pattern(imgSaida, P=24, R=3, method='uniform')
    hist = histograma(imgSaida)
    return hist

def extraiFileName(caminho):
    pedacos = caminho.split('\\')
    tamanho = len(pedacos)
    strSaida = pedacos[tamanho-1]
    return strSaida

def buscaDataset(nome_arquivo, dados):
    for x in dados:
        if nome_arquivo == x[0]: #se achou retorna a classificação
            if x[1] == 'positive': return 1
            elif x[1] == 'negative': return 0
    return False

def F1_score(revocacao, precisao):
    return 2*(revocacao*precisao)/(revocacao+precisao)

def novoAvaliaClassificador(y_original, y_previsto):
    falsoPositivo = 0
    verdadeiroPositivo = 0
    falsoNegativo = 0
    verdadeiroNegativo = 0
    for x in range(y_original.shape[0]):
        if y_original[x] == 0:
            if y_previsto[x] == 0:
                verdadeiroNegativo = verdadeiroNegativo + 1
            else:
                falsoNegativo = falsoNegativo + 1
        if y_original[x] == 1:
            if y_previsto[x] == 1:
                verdadeiroPositivo = verdadeiroPositivo + 1
            else:
                falsoPositivo = falsoPositivo + 1
    
    return falsoPositivo, verdadeiroPositivo, falsoNegativo, verdadeiroNegativo

def formataSaida(valor):
    saidaFormatada = "{:.2f}".format(valor*100)
    return saidaFormatada + "%"

def avalia_classificador_mais_metricas(clf, kf, X, y, f_metrica):
    metrica_val = []
    metrica_train = []
    precisao_val = []
    revocacao_val = []
    precisao_treino = []
    revocacao_treino = []
    f1_score = []
    for train, valid in kf.split(X,y):
        x_train = X[train]
        y_train = y[train]
        x_valid = X[valid]
        y_valid = y[valid]
        clf.fit(x_train, y_train)
        y_pred_val = clf.predict(x_valid)
        y_pred_train = clf.predict(x_train)
        metrica_val.append(f_metrica(y_valid, y_pred_val))
        FP_treino, VP_treino, FN_treino, VN_treino = novoAvaliaClassificador(y_train, y_pred_train)
        FP_val, VP_val, FN_val, VN_val = novoAvaliaClassificador(y_valid, y_pred_val)
        metrica_train.append(f_metrica(y_train, y_pred_train))
        precisao_treino.append(VP_treino / (VP_treino + FP_treino))
        revocacao_treino.append(VP_treino / (VP_treino + FN_treino))
        precisao_val.append(VP_val / (VP_val + FP_val))
        revocacao_val.append(VP_val / (VP_val + FN_val))
        f1_score.append(F1_score((VP_treino / (VP_treino + FN_treino)), (VP_treino / (VP_treino + FP_treino))))
    return np.array(metrica_val).mean(), np.array(metrica_train).mean(), np.array(precisao_treino).mean(), np.array(revocacao_treino).mean(), np.array(precisao_val).mean(), np.array(revocacao_val).mean(), np.array(f1_score).mean()

def avalia_classificador(clf, kf, X, y, f_metrica):
    metrica_val = []
    metrica_train = []
    for train, valid in kf.split(X,y):
        x_train = X[train]
        y_train = y[train]
        x_valid = X[valid]
        y_valid = y[valid]
        clf.fit(x_train, y_train)
        y_pred_val = clf.predict(x_valid)
        y_pred_train = clf.predict(x_train)
        metrica_val.append(f_metrica(y_valid, y_pred_val))
        metrica_train.append(f_metrica(y_train, y_pred_train))
    return np.array(metrica_val).mean(), np.array(metrica_train).mean()

def apresenta_metrica(nome_metrica, metrica_val, metrica_train, percentual = False):
    c = 100.0 if percentual else 1.0
    print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))
    print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))

def rodadaUnica(clf, X, y, f_metrica):
    clf.fit(X, y)
    y_pred_train = clf.predict(X)
    FP_treino, VP_treino, FN_treino, VN_treino = novoAvaliaClassificador(y, y_pred_train)
    metrica_train = (f_metrica(y, y_pred_train))
    precisao_treino = (VP_treino / (VP_treino + FP_treino))
    revocacao_treino = (VP_treino / (VP_treino + FN_treino))
    #print(f"F1-Score = {F1_score(precisao_treino, revocacao_treino)}")
    f1_score = F1_score(precisao_treino, revocacao_treino)
    return metrica_train, precisao_treino, revocacao_treino, y_pred_train, f1_score


## OPERACOES
df = pd.read_csv('E:\\Doutorado\\Aulas_Notas e Videos\\PDI\\Exercicios\\Trabalho Final\\Pulmao\\resultado.csv', header=1)
dataset = df.to_numpy()
dir_name = 'E:\\Doutorado\\Aulas_Notas e Videos\\PDI\\Exercicios\\Trabalho Final\\Pulmao\\reduzidas'
arquivo_dataset = 'E:\\Doutorado\\Aulas_Notas e Videos\\PDI\\Exercicios\\Trabalho Final\\Pulmao\\dataset.csv'

if (os.path.exists(arquivo_dataset)):
    dados_de_entrada = np.array(pd.read_csv(arquivo_dataset, header=1))
else:
    list_of_files = sorted( filter( os.path.isfile,
                            glob.glob(dir_name + '/**/*', recursive=True) ) )
    # Iterate over sorted list of files and print the file paths 
    # one by one.
    dados_de_entrada = []
    for file_path in list_of_files:
        #
        print(file_path)
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = skimage.transform.resize(img, (1024, 1024), mode='constant')
        processada = processamento(img)
        nome_arquivo = extraiFileName(file_path)
        classificacao = buscaDataset(nome_arquivo, dataset)
        processada.append(classificacao)
        dados_de_entrada.append(processada)
    
    dados_de_entrada = np.array(dados_de_entrada)
    
    #Salvando DATASET em arquivo
    pd.DataFrame(dados_de_entrada).to_csv(arquivo_dataset)

# Random... importante
np.random.shuffle(dados_de_entrada)

treino_size = int(np.floor((0.8 * len(dados_de_entrada))))
valid_size = len(dados_de_entrada) - int(np.floor((0.8 * len(dados_de_entrada))))

treino_set = dados_de_entrada[0:treino_size,:]
valid_set = dados_de_entrada[treino_size:,:]

coluna = dados_de_entrada.shape[1] - 1
#Normalizacao
treino_norm_X = normalizacao01_X(treino_set[:,:coluna])
valid_set_X = normalizacao01_X(valid_set[:,:coluna])

# Adição da Coluna 1
#treino_norm_X = np.c_[np.ones(treino_norm_X.shape[0]), treino_norm_X]
#teste_set_X = np.c_[np.ones(teste_set_X.shape[0]), teste_set_X]
#valid_set_X = np.c_[np.ones(valid_set_X.shape[0]), valid_set_X]

#Y's
Y_treino = treino_set[:,coluna]
Y_valid = valid_set[:,coluna]

################################
# Execução do Machine Learning #
################################

## Kfold padrão para todos os MLA
kf = KFold(n_splits=10, shuffle=True, random_state=10)

## SVM
from sklearn import svm

gammas = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 2, 3]
Cs = np.arange(-5, 17, 2, dtype=float)
best_acc = 0
print("Iniciando SVM")
for Ci in Cs:
    for gam in gammas:
        svc = svm.SVC(kernel="rbf", gamma=2**gam, C=2**Ci) # rbf é o default

        media_acuracia_val, media_acuracia_train = avalia_classificador(svc, kf, treino_norm_X, Y_treino, accuracy_score) 
        
        if media_acuracia_val > best_acc:
            best_acc = media_acuracia_val
            best_train = media_acuracia_train
            best_c = Ci
            best_g = gam

print('melhor resultado: Gamma={}, C = {}, acuracia = {}, treinamento={}'.format(best_g, best_c, best_acc, best_train))
# Rodada com o Melhor
best_svc = svm.SVC(kernel="rbf", gamma=2**best_g, C=2**best_c)
media_acuracia_val, media_acuracia_train = avalia_classificador(best_svc, kf, treino_norm_X, Y_treino, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

#Validação
media_acuracia_val, media_acuracia_train = avalia_classificador(best_svc, kf, valid_set_X, Y_valid, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

metrica_SVM, precisao_SVM, revocacao_SVM, y_SVM, f1_score_SVM = rodadaUnica(best_svc, valid_set_X, Y_valid, accuracy_score)
metrics.plot_roc_curve(best_svc, valid_set_X, Y_valid)
disp = plot_precision_recall_curve(best_svc, valid_set_X, Y_valid)
disp.ax_.set_title('Precision-Recall Binária SVM')
print(f"F1-Score: {formataSaida(f1_score_SVM)}")
print(f"Acurácia: {formataSaida(metrica_SVM)}")
print(f"Precisao: {formataSaida(precisao_SVM)}")
print(f"Revocacao: {formataSaida(revocacao_SVM)}")

plt.show() 


## ADABOOST
   
from sklearn.ensemble import AdaBoostClassifier

best_acc = 0
print("Iniciando ADABOOST")
for n in [50, 200, 500]:
    for l in [1, 0.5, 0.3, 0.2]:
#        print('n_estimators={}, learning_rate = {}'.format(n, l))
        ada = AdaBoostClassifier(n_estimators=n, learning_rate=l)
        media_acuracia_val, media_acuracia_train = avalia_classificador(ada, kf, treino_norm_X, Y_treino, accuracy_score) 
        #apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)
        if media_acuracia_val > best_acc:
            best_acc = media_acuracia_val
            best_train = media_acuracia_train
            best_n = n
            best_l = l
print('melhor resultado: n_estimators={}, learning_rate = {}, acuracia = {}, treinamento={}'.format(best_n, best_l, best_acc, best_train))

best_ada = AdaBoostClassifier(n_estimators=best_n, learning_rate=best_l)

media_acuracia_val, media_acuracia_train = avalia_classificador(best_ada, kf, treino_norm_X, Y_treino, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

#Validação
media_acuracia_val, media_acuracia_train = avalia_classificador(best_ada, kf, valid_set_X, Y_valid, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

metrica_ADA, precisao_ADA, revocacao_ADA, y_ADA, f1_score_ADA = rodadaUnica(best_ada, valid_set_X, Y_valid, accuracy_score)
metrics.plot_roc_curve(best_ada, valid_set_X, Y_valid)
disp = plot_precision_recall_curve(best_ada, valid_set_X, Y_valid)
disp.ax_.set_title('Precision-Recall Binária ADABoost')
print(f"F1-Score: {formataSaida(f1_score_ADA)}")
print(f"Acurácia: {formataSaida(metrica_ADA)}")
print(f"Precisao: {formataSaida(precisao_ADA)}")
print(f"Revocacao: {formataSaida(revocacao_ADA)}")

## Regressão logística

from sklearn.linear_model import LogisticRegression

print("Iniciando Regressão Logistica")
lr = LogisticRegression(solver='liblinear')

media_acuracia_val, media_acuracia_train = avalia_classificador(lr, kf, treino_norm_X, Y_treino, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

metrica_LR, precisao_LR, revocacao_LR, y_LR, f1_score_LR = rodadaUnica(lr, valid_set_X, Y_valid, accuracy_score)
metrics.plot_roc_curve(lr, valid_set_X, Y_valid)
disp = plot_precision_recall_curve(lr, valid_set_X, Y_valid)
disp.ax_.set_title('Precision-Recall Binária Regressão Logistica')
print(f"F1-Score: {formataSaida(f1_score_LR)}")
print(f"Acurácia: {formataSaida(metrica_LR)}")
print(f"Precisao: {formataSaida(precisao_LR)}")
print(f"Revocacao: {formataSaida(revocacao_LR)}")

#Arvore de Decisão

from sklearn import tree
dt = tree.DecisionTreeClassifier(max_depth=3)

print("Iniciando Árvore de Decisão")

media_acuracia_val, media_acuracia_train = avalia_classificador(dt, kf, treino_norm_X, Y_treino, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

metrica_Tree, precisao_Tree, revocacao_Tree, y_Tree, f1_score_Tree = rodadaUnica(dt, valid_set_X, Y_valid, accuracy_score)

metrics.plot_roc_curve(dt, valid_set_X, Y_valid)
disp = plot_precision_recall_curve(dt, valid_set_X, Y_valid)

disp.ax_.set_title('Precision-Recall Binária Árvore de Decisão')
print(f"F1-Score: {formataSaida(f1_score_Tree)}")
print(f"Acurácia: {formataSaida(metrica_Tree)}")
print(f"Precisao: {formataSaida(precisao_Tree)}")
print(f"Revocacao: {formataSaida(revocacao_Tree)}")

import graphviz 
dot_data = tree.export_graphviz(dt, out_file=None, 
                                feature_names=['0','1', '2', '3', '4', '5','6','7',
                                               '8','9', '10', '11', '12', '13',
                                               '14','15','16', '17', '18', '19', 
                                               '20','21', '22', '23', '24', '25', '26'],  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

## Classificador MLP
from sklearn.neural_network import MLPClassifier

print("Iniciando MLP")

best_acc = 0
for alpha in [1e-2, 1e-3, 1e-4, 1e-5]:
    for hidden in [64, 128, 256]:#, 10, 12]:
        mlp = MLPClassifier(solver='lbfgs', alpha=alpha, 
                            hidden_layer_sizes=(hidden, 2), random_state=1, max_iter=2000)
        
        media_acuracia_val, media_acuracia_train = avalia_classificador(mlp, kf, treino_norm_X, Y_treino, accuracy_score)
        #apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)
        if media_acuracia_val > best_acc:
            best_acc = media_acuracia_val
            best_train = media_acuracia_train
            best_a = alpha
            best_h = hidden

best_mlp = MLPClassifier(solver='lbfgs', alpha=best_a, 
                            hidden_layer_sizes=(best_h, 2), random_state=1)
        
media_acuracia_val, media_acuracia_train = avalia_classificador(best_mlp, kf, treino_norm_X, Y_treino, accuracy_score)
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

#Validação
media_acuracia_val, media_acuracia_train = avalia_classificador(best_mlp, kf, valid_set_X, Y_valid, accuracy_score) 
apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)

metrica_MLP, precisao_MLP, revocacao_MLP, y_MLP, f1_score_MLP = rodadaUnica(best_mlp, valid_set_X, Y_valid, accuracy_score)

metrics.plot_roc_curve(best_mlp, valid_set_X, Y_valid)
disp = plot_precision_recall_curve(best_mlp, valid_set_X, Y_valid)

disp.ax_.set_title('Precision-Recall Binária MLP')
print(f"F1-Score: {formataSaida(f1_score_MLP)}")
print(f"Acurácia: {formataSaida(metrica_MLP)}")
print(f"Precisao: {formataSaida(precisao_MLP)}")
print(f"Revocacao: {formataSaida(revocacao_MLP)}")

# Commitee
comite = np.concatenate((y_MLP.reshape((y_MLP.shape[0],1)) , y_LR.reshape((y_LR.shape[0],1))), axis=1)
comite = np.concatenate((comite , y_LR.reshape((y_LR.shape[0],1))), axis=1)
comite = np.concatenate((comite , y_Tree.reshape((y_Tree.shape[0],1))), axis=1)
comite = np.concatenate((comite , y_ADA.reshape((y_ADA.shape[0],1))), axis=1)

saida_comite = np.zeros(len(comite))
democracia = np.ceil(comite.shape[1]/2)
for i in range(len(comite)):
    if np.sum(comite[i]) >= democracia:
        saida_comite[i] = 1
    else:
        saida_comite[i] = 0
        
    
FP_comite, VP_comite, FN_comite, VN_comite = novoAvaliaClassificador(saida_comite, Y_valid)
                        
precisao = VP_comite / (VP_comite + FP_comite)
revocacao = VP_comite / (VP_comite + FN_comite)
f1_score = F1_score((VP_comite / (VP_comite + FN_comite)), (VP_comite / (VP_comite + FP_comite)))
                        
                       
print(f"F1-Score: {formataSaida(f1_score)}")
#print(f"Acurácia: {formataSaida(metrica_MLP)}")
print(f"Precisao: {formataSaida(precisao)}")
print(f"Revocacao: {formataSaida(revocacao)}")                   
                        

# Gráfico
from sklearn.model_selection import ShuffleSplit

# Definição do gráfico
fig, axes = plt.subplots(3, 5, figsize=(10, 15))

cross = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

plot_learning_curve(best_svc, "SVM Learning Curves", treino_norm_X, Y_treino, axes=None, ylim=None, cv=cross, n_jobs=None)

plot_learning_curve(dt, "Decision Tree Learning Curves", treino_norm_X, Y_treino, axes=None, ylim=None, cv=cross, n_jobs=None)

plot_learning_curve(lr, "Linear Regression Learning Curves", treino_norm_X, Y_treino, axes=None, ylim=None, cv=cross, n_jobs=None)

plot_learning_curve(best_ada, "ADABoost Learning Curves", treino_norm_X, Y_treino, axes=None, ylim=None, cv=cross, n_jobs=None)

plot_learning_curve(best_mlp, "MLP Learning Curves", treino_norm_X, Y_treino, axes=None, ylim=None, cv=cross, n_jobs=None)

plt.show()

