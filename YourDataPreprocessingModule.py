import pyodbc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pickle

data_f = pd.read_csv('dataset/flexdb.csv')
data_m = pd.read_csv('dataset/dbcleaned2.csv')
data = pd.read_csv('dataset/izodedb.csv')

#function to preprocess data for izod prediction
def perform_one_hot_encoding2(input_data):
    cat_cols = ['Items','COULEUR' ,'I_CM','I_G', 'I_F']
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(data[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    new_inputs_encoded = encoder.transform(input_data[cat_cols])
    new_inputs_encoded = pd.DataFrame(new_inputs_encoded, columns=encoder.get_feature_names_out(cat_cols))
    input_data_df = pd.DataFrame(input_data, columns=['Items', 'melt','flex', 'traction','COULEUR', 'I_CM', 'I_G', 'I_F', 'cendre', 'I1'])
    input_data_encoded = pd.concat([input_data_df, new_inputs_encoded], axis=1)
    input_data_encoded = input_data_encoded.drop(cat_cols, axis=1)   
    return input_data_encoded

#function to preprocess data for flexion prediction
def perform_one_hot_encoding(input_data,encoder):
    cat_cols = ['Items','COULEUR' ,'I_CM','I_G', 'I_F']
    new_inputs_encoded = encoder.transform(input_data[cat_cols])
    new_inputs_encoded = pd.DataFrame(new_inputs_encoded, columns=encoder.get_feature_names_out(cat_cols))
    input_data_encoded = pd.concat([input_data.drop(cat_cols, axis=1), new_inputs_encoded], axis=1)
    return input_data_encoded

#function to preprocess data for melt prediction
def perform_one_hot_encoding_for_melt(input_data):
    cat_cols = ['Items','COULEUR' ,'I_CM','I_G', 'I_F']
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(data_m[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    new_inputs_encoded = encoder.transform(input_data[cat_cols])
    new_inputs_encoded = pd.DataFrame(new_inputs_encoded, columns=encoder.get_feature_names_out(cat_cols))
    input_data_df = pd.DataFrame(input_data, columns=['Items', 'densite','COULEUR', 'I_CM', 'I_G', 'I_F'])
    input_data_encoded = pd.concat([input_data_df, new_inputs_encoded], axis=1)
    input_data_encoded = input_data_encoded.drop(cat_cols, axis=1)
    return input_data_encoded

#fonctions to preprocess data for mixture prediction

def topourcentage(df):
    #add a column 'sum' that sums the values of the column 'Poids'
    df['Sum'] = df['Poids'].sum()
    #add a column 'pourcentage' that calculates the pourcentage in  float .2f of each component  
    df['Pourcentage'] = df['Poids'] / df['Sum']
    #round the values of the column 'pourcentage' to 2 decimals
    df['Pourcentage'] = df['Pourcentage'].apply(lambda x: round(x, 2))
    return df

def toarray(df):
    #add the values of the column 'pourcentage' in the list 'array'
    df['array'] = df[['Items','Melt','Pourcentage']].values.tolist()
    return df

def melt_theo(df):
    df['Log_melt'] = np.log10(df['Melt'])
    #new column log_melt * pourcentage
    df['Log_melt * Pourcentage'] = df['Log_melt'] * df['Pourcentage']
    #new column 'SumX' that sums the values of the column 'log_melt * pourcentage' 
    df['SumX'] = df['Log_melt * Pourcentage'].sum()
    #new column named melt_theo that contains of the values of the column 'sum_log_melt * pourcentage' to the power of 10
    df['Melt_theo'] = 10**df['SumX']
    return df

def todatasetbrut(df):
    #create a new dataframe with 10 columns named 'comp1' to 'comp10'
    df2 = pd.DataFrame(columns=['comp1','comp2','comp3','comp4','comp5','comp6','comp7','comp8','comp9','comp10'])  
    array_values = df['array'].values.tolist()
    # Fill remaining rows with N    
    for i in range(10):
            if i < len(df):
                    df2.loc[0, f'comp{i+1}'] = array_values[i-1]
            else:
                df2.loc[0, f'comp{i+1}'] = [0,0,0]
    #return only first row of the dataframe df2
    df2['melt_theorique']= df['Melt_theo']
    return df2

def todataset(df,nb_composants):
    num_comps = nb_composants
    X = []
    for  row in df.iterrows():
       for index, row in df.iterrows():
        row_data = []
        for i in range(1, num_comps + 1):
            cell = row[f"comp{i}"]
            processed_cell = cell
            if processed_cell is not [0,0,0]:
                row_data.append(processed_cell)
        X.append(row_data)

    # Normaliser les données d'entrée
    scaler = MinMaxScaler()
    X_scaled = [scaler.fit_transform(seq) for seq in X]

    # Ajuster la taille des séquences (remplissage avec des zéros)
    max_seq_length = max(len(seq) for seq in X_scaled)
    X_padded = np.zeros((len(X_scaled), max_seq_length, 3))

    for i, seq in enumerate(X_scaled):
        X_padded[i, :len(seq)] = seq
    
    return [X_padded]


#fonction to preprocess data for mixture prediction
def perform_melt_index_preprocessing(input_data):
    df_Pourcentage = topourcentage(input_data)
    df_array = toarray(df_Pourcentage)
    df_melt_theo = melt_theo(df_array)
    df_dataset = todatasetbrut(df_melt_theo)
    return df_dataset
#lstm model prediction
def pred(X,model):   
    y = model.predict(X)   
    return y

#function to extract data from access 
def extract_data_from_access(query):
    # create the connection string using the ODBC driver name and database path
    driver = 'Microsoft Access Driver (*.mdb, *.accdb)'
    database_path = r'C:\Users\Ilyas\OneDrive\Bureau\Udes-Projet\GI-HISTO.mdb'
    cnxn_str = f"DRIVER={{{driver}}};DBQ={database_path}"

    # create the pyodbc connection object
    cnxn = pyodbc.connect(cnxn_str)
    
    # use Pandas to read data directly from the Access database
    df = pd.read_sql(query, cnxn)

    # close the connection object
    cnxn.close()

    return df


#function to convert columns to numerical if it is possible
def convert_to_numerical(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df
#function to merge two dataframes
def merge(df1, df2, column):
    df = pd.merge(df1, df2, on=column)
    return df

#function to drop columns
def drop(df, columns):
    df = df.drop(columns, axis=1)
    return df

def dropnaand0(df, column):
    df = df.dropna(subset=[column])
    df = df[df[column] != 0]
    df = df[df[column] != '0']
    return df

#function that estimate missing values in a dataframe with machine learning
def estimate_missing_values_ml(df, columns):
    #df with columns 
    df = df[columns]
    #create a list of columns with missing values
    missing_values = df.columns[df.isnull().any()]
    #loop through columns with missing values
    for col in missing_values:
        #create a dataframe with the column to estimate and the columns without missing values
        df_temp = df[[col] + list(df.columns[df.columns != col])]
        #split the data into train and test
        X_train = df_temp[df_temp[col].notnull()].drop(col, axis=1)
        y_train = df_temp[df_temp[col].notnull()][col]
        X_test = df_temp[df_temp[col].isnull()].drop(col, axis=1)
        #fit the model to the training data
        model = RandomForestRegressor().fit(X_train, y_train)
        #predict the missing values
        df.loc[df[col].isnull(), col] = model.predict(X_test)
    return df

#defining a function to find the non numeric values in a column
def Findnonnumeric(df,colomn):
    # Find non-numeric strings in a single column
    a = df[colomn]
    msk = a.notnull() & (a != 0)
    a = a[msk]
    non_numeric_strings = a[pd.to_numeric(a, errors='coerce').isna()]
    # Display the non-numeric strings
    print(f"Non-numeric strings in column {colomn}:")
    print(non_numeric_strings)
    print('--------------------------------------------------------------------------')

