from flask import Flask, request, render_template
from sqlalchemy import create_engine
import psycopg2
import psycopg2.extras
import pandas as pd
import json

app = Flask(__name__)

# BACK-END
# Load database settings
with open('db_settings.json', 'r') as f:
    db_settings = json.load(f)

# Database connection function with psycopg2
def connect_from_settings(settings):
    return psycopg2.connect(
        dbname=settings['pgdb'],
        user=settings['pguser'],
        password=settings['pgpasswd'],
        host=settings['pghost'],
        port=settings['pgport']
    )

# SQLalchemy
def get_engine(user, passwd, host, port, db, schema):
    """
    Get SQLalchemy engine using credentials.
    Input:
        db: database name
        user: Username
        host: Hostname of the database server
        port: Port number
        passwd: Password for the database
        schema: Database schema
    Returns:
        Database engine
    """

    url = 'postgresql://{user}:{passwd}@{host}:{port}/{db}'.format(
        user=user, passwd=passwd, host=host, port=port, db=db)
    engine = create_engine(url,connect_args={'options' : f'--search_path={schema}'}, pool_size=50, echo=False)
    return engine

def get_engine_from_settings(settings):
    """
    Sets up database connection from local settings.
    Input:
        settings: Dictionary containing pghost, pguser, pgpassword, pgdatabase, pgport and schema.
    Returns:
        Call to get_database returning engine
    """
    keys = ['pguser','pgpasswd','pghost','pgport','pgdb','schema']
    if not all(key in keys for key in settings.keys()):
        raise Exception('Bad config file')

    return get_engine(settings['pguser'],
                      settings['pgpasswd'],
                      settings['pghost'],
                      settings['pgport'],
                      settings['pgdb'],
                      settings['schema'])


# FRONT-END
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/injection', methods=['GET', 'POST'])
def injection():
    if request.method == 'POST':
        loc = request.form['location']
        query = f'SELECT location, café, drive_thru FROM branch WHERE location LIKE \'%{loc}%\';'
        connection = connect_from_settings(db_settings)
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query)
        dict_res = cursor.fetchall()
        df = pd.DataFrame(dict_res,columns=list(dict_res[0].keys()))
        cursor.close()
        connection.close()
        return render_template('injection.html', table=df.to_html(), query=query)
    else:
        return render_template('injection.html')
    
@app.route('/sanitized', methods=['GET', 'POST'])
def sanitized():
    title = "Sanitized Page"
    if request.method == 'POST':
        loc = request.form['location']
        query = 'SELECT location, café, drive_thru FROM branch WHERE location LIKE %s;'
        connection = connect_from_settings(db_settings)
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, ("%" + loc + "%",))  # We add the wildcards (%) so it can still return results for not-exact matches
        dict_res = cursor.fetchall()
        cursor.close()
        connection.close()
        if len(dict_res) > 0:
            df = pd.DataFrame(dict_res, columns=list(dict_res[0].keys()))
            return render_template('sanitized.html', table=df.to_html(), query=query, error=None, title=title)
        else:
            error = "Could not find anything that matches your query"
            return render_template('sanitized.html', table=None, query=query, error=error, title=title)
    else:
        return render_template('sanitized.html', title=title)

@app.route('/parametrized', methods=['GET', 'POST'])
def parametrized():
    title = "Parametrized Page"
    if request.method == 'POST':
        loc = request.form['location']
        query = 'SELECT location, café ,drive_thru FROM branch WHERE location LIKE %(location)s;'
        connection = connect_from_settings(db_settings)
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query,{"location":"%"+loc+"%"}) # We add the wildcards (%) so it can still return results for not-exact matches
        dict_res = cursor.fetchall()
        cursor.close()
        connection.close()
        if len(dict_res) > 0:
            df = pd.DataFrame(dict_res, columns=list(dict_res[0].keys()))
            return render_template('sanitized.html', table=df.to_html(), query=query, error=None, title=title)
        else:
            error = "Could not find anything that matches your query"
            return render_template('sanitized.html', table=None, query=query, error=error, title=title)
    else:
        return render_template('sanitized.html', title=title)
    
@app.route('/alchemy', methods=['GET', 'POST'])
def alchemy():
    title = "SQL Alchemy back-end"
    if request.method == 'POST':
        loc = request.form['location']
        query = "SELECT location,café,drive_thru FROM branch WHERE location LIKE %(location)s;"
        engine = get_engine_from_settings(db_settings)
        df = pd.read_sql_query(query,engine,params={"location":"%"+loc+"%"})
        engine.dispose()
        return render_template('sanitized.html', table=df.to_html(), query=query, error=None, title=title)
    else:
        return render_template('sanitized.html', title=title)

###----------------------------------------------------------------------------------------------------------------------------------###
# Homework 3.
@app.route('/menu', methods=['GET', 'POST'])
def menu():
    title = "Menu Search"
    if request.method == 'POST':
        item_name = request.form.get('item_name', '')
        min_price = request.form.get('min_price', None)
        max_price = request.form.get('max_price', None)
        item_type = request.form.get('item_type', '')

        # Constructing the base query
        query = """
            SELECT item_id, item_name, item_price, item_type 
            FROM menu
            WHERE 1=1
        """ ### 1=1 is required at the start of the following filters, followed by " AND ..."
        params = {}

        # Adding filters if they are provided      
        if item_name:
            query += " AND item_name LIKE %(item_name)s;"
            params['item_name'] = '%' + item_name + '%'
        
        if min_price:
            query += " AND item_price >= %(min_price)s;"
            params['min_price'] = float(min_price)
        
        if max_price:
            query += " AND item_price <= %(max_price)s;"
            params['max_price'] = float(max_price)
        
        if item_type:
            query += " AND %(item_type)s = ANY(item_type);"
            params['item_type'] = item_type
        
        # If no params are given, return with error
        if not params:
            error = "Fill at least one cell."
            return render_template('menu.html', table=None, query=query, error=error, title=title)            

        connection = connect_from_settings(db_settings)
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, params)
        dict_res = cursor.fetchall()
        cursor.close()
        connection.close()
        if dict_res:
            df = pd.DataFrame(dict_res, columns=list(dict_res[0].keys()))
            return render_template('menu.html', table=df.to_html(), query=query, error=None, title=title)
        else:
            error = "No menu items match your filters."
            return render_template('menu.html', table=None, query=query, error=error, title=title)
    else:
        return render_template('menu.html', title=title)
# end of Homework 3.
###----------------------------------------------------------------------------------------------------------------------------------###
    

if __name__ == '__main__':
    app.run(debug=True)