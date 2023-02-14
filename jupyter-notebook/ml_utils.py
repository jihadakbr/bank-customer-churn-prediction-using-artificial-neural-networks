## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

## for statistical tests
import scipy
import scipy.stats as stats

## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble
import imblearn

## for deep learning
from tensorflow.keras import models, layers, backend as K

## save file/model
import joblib

###############################################################################
###############################################################################


def filter_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    upper_limit = q3 + 1.5 * (q3-q1)
    lower_limit = q1 - 1.5 * (q3-q1)

    data = data[data[column] > lower_limit]
    data = data[data[column] < upper_limit]
    
    return data

def outliers_subplot(data,col1,col2,col3,col4):
    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    fig.tight_layout(w_pad=5.0)

    sns.set_style("whitegrid")
    custom_palette = sns.color_palette("muted")

    sns.boxplot(y=data[col1], ax=ax[0], color=custom_palette[0])
    sns.boxplot(y=data[col2], ax=ax[1], color=custom_palette[1])
    sns.boxplot(y=data[col3], ax=ax[2], color=custom_palette[2])
    sns.boxplot(y=data[col4], ax=ax[3], color=custom_palette[3])
  
    ax[0].set_ylabel("Credit Score", fontsize=14)
    ax[1].set_ylabel("Age", fontsize=14)
    ax[2].set_ylabel("Balance", fontsize=14)
    ax[3].set_ylabel("Estimated Salary", fontsize=14)

    for i in range(4):
        ax[i].grid(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
    
    return fig, ax

def num_dist(data):
    var_group = data.columns
    plt.figure(figsize=(12,7), dpi=400)

    for j,i in enumerate(var_group):
        
        mini = data[i].min()
        maxi = data[i].max()
        ran = data[i].max() - data[i].min()
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        points = mean-st_dev, mean+st_dev

        plt.subplot(2,2,j+1)
        
        sns.distplot(data[i], hist=True, kde=True)
        
        sns.lineplot(x=points, y=[0,0], color='black', label="std_dev")
        sns.scatterplot(x=[mini,maxi], y=[0,0], color='orange', label="min/max")
        sns.scatterplot(x=[mean], y=[0], color='red', label="mean")
        sns.scatterplot(x=[median], y=[0], color='blue', label="median")
        plt.xlabel('{}'.format(i), fontsize=20)
        plt.ylabel('density')
        plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),round(kurt,2),round(skew,2),(round(mini,2),round(maxi,2),round(ran,2)),round(mean,2),round(median,2)))
        sns.despine(top=True, right=True)
        ymin, ymax = plt.gca().get_ylim()
        plt.grid(False)
        plt.ylim(min(ymin, -0.05 * ymax), ymax)
    plt.tight_layout()

def target_dist(data,col,label1,label2):
    mpl.rcParams['font.size'] = 11
    r = data.groupby(col)[col].count()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(r, explode=[0.05, 0.1], labels=[label1, label2], radius=1.5, autopct='%1.1f%%', shadow=True, startangle=45,
           colors=['#66b3ff', '#ff9999'])
    ax.set_aspect('equal')
    ax.set_frame_on(False)

def cat_dist(data, x, hue, palette):
    sns.set_style("whitegrid")
    ax = sns.countplot(data=data, x=x, hue=hue, palette=palette)
    sns.despine(top=True, right=True)
    plt.xlabel(f"{x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.grid(False)

    for i in ax.patches:
        ax.text(i.get_x()+0.1, i.get_height()+50, int(i.get_height()), fontsize=11)
        
def num_corr(data):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    correlation = data.corr()
    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlation, annot=True, mask=mask, cmap='coolwarm', annot_kws={"size": 11})
    sns.despine(left=True, bottom=True)
    plt.grid(False)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title("Correlation Matrix", fontsize=15, fontweight='bold')
    
def chi_square_test(data, col_x, col_target):
    crosstab = pd.crosstab(data[col_x], data[col_target])
    stat, p, dof, expected = stats.chi2_contingency(crosstab)
    alpha = 0.05
    print("p-value is: ", p)
    if p <= alpha:
        print('Dependent (reject H0)\nThis feature is dependent on the target variable.')
    else:
        print('Independent (H0 holds true)\nThis feature is independent on the target variable.')
              
def stack_hist_ct(data,col_x,col_target,color_col_x,color_col_target,label1,label2):
    ct_geo_exit = pd.crosstab(data[col_x], data[col_target])
    ax = ct_geo_exit.plot(kind="bar", stacked=True, color=[color_col_x, color_col_target])
    plt.xlabel(f"{col_x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([label1, label2])
    sns.despine(top=True, right=True)
    plt.grid(False)
    for i in ax.containers:
        ax.bar_label(i)

    plt.xticks(rotation=0)        
    
def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))    
    
def plot_confusion_matrix(y_test, y_pred, cmap='Reds'):
    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=['Predicted Not Exited', 'Predicted Exited'], 
                yticklabels=['Actual Not Exited', 'Actual Exited'])
    plt.title('Confusion Matrix')
    plt.show()    
 
def plot_metrics(training, metric_1, metric_2):
    plt.plot(np.array(training.history[metric_1]) * 100)
    plt.plot(np.array(training.history[metric_2]) * 100)
    plt.ylabel(metric_1)
    plt.xlabel('epochs')
    plt.legend(['train', 'validation'])
    sns.despine(top=True, right=True)    
    plt.grid(False)
    plt.show()    