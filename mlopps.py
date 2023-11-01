import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pymysql

def create_connection():
    ''' This function creates a database connection   '''
    conn = pymysql.connect(host='localhost', 
                           port=3306, 
                           user='root', 
                           password='12qwaszx', 
                           db='housing')
    return conn

def execute_query(query, cur, conn):
    '''function to connect and execute query; if fails, it rollsback'''
    try:
        cur.execute(query)
        conn.commit()
    except Exception as ex:
        print(ex)
        conn.rollback() 


def create_tables(cur,conn):
    '''function to create table with housing data '''
    create_housing_data_table = '''CREATE TABLE IF NOT EXISTS housing_data (
        housing_id INT AUTO_INCREMENT PRIMARY KEY,
        price INT,
        area INT,
        bedrooms TINYINT,
        bathrooms TINYINT,
        stories NVARCHAR(50),
        mainroad NVARCHAR(50),
        guestroom NVARCHAR(50),
        basement NVARCHAR(50),
        hotwaterheating NVARCHAR(50),
        airconditioning NVARCHAR(50),
        parking NVARCHAR(50),
        prefarea NVARCHAR(50),
        furnishingstatus ENUM ('furnished','semi-furnished','unfurnished')
    );'''
    execute_query(create_housing_data_table,cur,conn)
    

def read_data():
    '''function to read in the csv'''
    housing_csv = pd.read_csv('Housing.csv')
    return housing_csv


def load_data(cur,conn):
    '''function to load the csv into the database table'''
    load_data = '''LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Housing.csv'
    INTO TABLE housing_data
    FIELDS TERMINATED BY ','
    ENCLOSED BY '"'
    LINES TERMINATED BY '\r\n'
    IGNORE 1 ROWS
    (price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus)'''
    execute_query(load_data,cur,conn)

def make_dataframe(conn):
    '''function to make data into a dataframe'''
    make_dataframe = '''SELECT * FROM housing_data'''

    df = pd.read_sql(make_dataframe, con= create_connection())

def data_cleaning(df):
    '''function to clean and prep data'''
    yes_no =['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for c in yes_no:
        df[c]= df[c].apply(lambda x: True if x=='yes' else False)
    mapping = {'yes': 1, 'no': 0}
    columns_to_map = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[columns_to_map]= df[columns_to_map].replace(mapping)
    df= pd.get_dummies(df, columns =['furnishingstatus'])
    
    return df

def split_data(df):
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test= train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def run_a_model( X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    MSE = mean_squared_error(y_test, predictions)
    print('MSE {}'.format(MSE))
    RMSE = mean_squared_error(y_test, predictions, squared = False)
    print('RMSE {}'.format(RMSE))
    MAE = mean_absolute_error(y_test, predictions)
    print('MAE {}'.format(MAE))
    r2 = r2_score(y_test, predictions)
    print('r2 {}'.format(r2))
    return MSE, RMSE, MAE, r2

def main():
    conn = create_connection()
    cur = conn.cursor()
    create_tables(cur,conn)
    # read_data()
    load_data(cur,conn)
    df = make_dataframe(conn)
    df =data_cleaning(df)
    X_train, X_test, y_train, y_test =split_data(df)
    MSE, RMSE, MAE, r2 =run_a_model(X_train, X_test, y_train, y_test)
    conn.close()

if __name__ == '__main__':
    main()