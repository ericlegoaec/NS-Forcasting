import numpy as np
import pandas as pd
from scipy.optimize import minimize

'''
def NelsonSiegelSvenssonParams(x,y):
 
  
    fp = lambda c, x: (c[0])+ (c[1]*((1- np.exp(-x/20))/(x/20)))+ (c[2]*((((1-np.exp(-x/20))/(x/20)))- (np.exp(-x/20))))+ (c[3]*((((1-np.exp(-x/10))/(x/10)))- (np.exp(-x/10))))
    
    # error function to minimize
    e = lambda p, x, y: ((fp(p,x)-y)**2).sum()
    p0 = np.array([0.01,0.01,0.01,0.01])    
    return (minimize(e, p0, args=(x,y)).x)
    
def evalNelsonSiegelSvensson(m,params):
    c = params
    j = []
    tun1 = 20
    tun2 = 10
    for h in m:
        #print(len(j))
        j.append((c[0])+ (c[1]*((1- np.exp(-h/tun1))/(h/tun1)))+ (c[2]*((((1-np.exp(-h/tun1))/(h/tun1)))- (np.exp(-h/tun1))))+ (c[3]*((((1-np.exp(-h/tun2))/(h/tun2)))- (np.exp(-h/tun2)))))
    return j
    
'''

def NelsonSiegelSvenssonParams(x,y):
    '''
    Inputs:
    parametric function, x is the independent variable (maturities)
    and c are the parameters.
    it's a polynomial of degree 2
    
    Outputs:
    returns the parameters for NSS

    '''
    fp = lambda c, x: (c[0])+ (c[1]*((1- np.exp(-x/c[4]))/(x/c[4])))+ (c[2]*((((1-np.exp(-x/c[4]))/(x/c[4])))- (np.exp(-x/c[4]))))+ (c[3]*((((1-np.exp(-x/c[5]))/(x/c[5])))- (np.exp(-x/c[5]))))
    
    # error function to minimize
    e = lambda p, x, y: ((fp(p,x)-y)**2).sum()
    p0 = np.array([0.01,0.01,0.01,0.01,1.00,1.00])    
    return (minimize(e, p0, args=(x,y)).x)

def evalNelsonSiegelSvensson(t,params):
    '''
    Inputs:
    m is a list of maturities at which to be evaluated at
    params are the parameters we are using to evaluate
    '''
    c = params
    j = []
    for h in t:
        #print(len(j))
        j.append((c[0])+ (c[1]*((1- np.exp(-h/c[4]))/(h/c[4])))+ (c[2]*((((1-np.exp(-h/c[4]))/(h/c[4])))- (np.exp(-h/c[4]))))+ (c[3]*((((1-np.exp(-h/c[5]))/(h/c[5])))- (np.exp(-h/c[5])))))
    return j




def getNSSParams(df):
    '''
    Inputs: 
    df is a dataframe of yield to be evaluated, rows=dates, columns=maturities
    
    Outputs:
    returns a dataframe containing the parameters of NSS for the yield curves of each dates
    '''
    x = df.columns.values
    dic={}
    for index, row in df.iterrows():
        y = df.loc[index].values
        params = NelsonSiegelSvenssonParams(x,y)
        dic[index] = params
    return pd.DataFrame.from_dict(dic)

def getPred(df,df_params):
    # for the future, pass in what function you want to use to evaluate error
    '''
    Inputs:
    df is the original dataframe containing the data
    df_params is the parameters containing the parameters used to predict
    
    Output:
    returns a dataframe of the predicted values formated like the original
    '''
    x = df.columns.values
    dic = {}
    for column in df_params:
        dic[column] = evalNelsonSiegelSvensson(x,df_params[column])
    predicted = pd.DataFrame.from_dict(dic).transpose()
    predicted.columns = x
    return predicted

def getPredYieldCurveError(df, df_pred):
    '''
    Input:
    df is the original dataframe containing the data
    df_pred is the predicted yield using the parameters
    
    Output:
    returns the error of the yield curve for each date
    '''
    df_error = df - df_pred
    _,n = df_error.shape
    dic = {}
    for index, row in df_error.iterrows():
        error = np.sqrt(np.sum(row.values**2)/n)
        dic[index] = error
    return dic

def getPredMatError(df, df_pred):
    '''
    Input:
    df is the original dataframe containing the data
    df_pred is the predicted yield using the parameters
    
    Output:
    returns the error of the yield grouped by maturity dates
    '''
    df_error = df - df_pred
    n,_ = df_error.shape
    dic = {}
    for column in df_error:
        error = np.sqrt(np.sum(df_error[column]**2)/n)
        dic[column] = error
    return dic




