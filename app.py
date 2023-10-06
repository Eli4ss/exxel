from flask import Flask, request, jsonify ,render_template,session, redirect, url_for,flash
from database import *
from YourDataPreprocessingModule import *
import pandas as pd
import pickle
from models import User  # Import your User model here
from keras.models import load_model
from random import randint
import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)


app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Set a secret key for session security


model_izod = pickle.load(open('models/extrmodelizode.pkl', 'rb'))
model_melt = pickle.load(open('models/rfmodel.pkl', 'rb'))


# Create a dictionary of allowed usernames and passwords
users = [User('admin', 'admin')]

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Find the user in the users list (replace with database query in production)
        user = next((u for u in users if u.username == username), None)

        if user and user.password == password:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = "Login failed. Please try again."

    return render_template('login.html', error=error)


@app.route('/chartjs')
def chartjs():
    return render_template('chartjs.html')
    


@app.before_request
def require_login():
    if not session.get('logged_in') and request.endpoint != 'login':
        return redirect('login')




@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect('login')



@app.route('/izod',methods=['GET', 'POST'])
def izod():
    predicted_izod = None  # Initialize predicted Izod value as None
    prediction = None  # Initialize prediction as None
    if request.method == 'POST':
        items = request.form.get('items')
        melt = float(request.form.get('melt'))
        couleur = request.form.get('couleur')
        i_cm = request.form.get('I_CM')
        i_g = request.form.get('I_G')
        i_f = request.form.get('I_F')
        cendre = float(request.form.get('cendre'))
        flexion = float(request.form.get('flexion'))
        traction = float(request.form.get('traction'))
        i1 = float(request.form.get('i1'))

        # Create a DataFrame from user inputs
        new_inputs_df = pd.DataFrame([[items,melt,flexion, traction, couleur, i_cm, i_g, i_f, cendre, i1]],
                                     columns=['Items', 'melt','flex', 'traction','COULEUR', 'I_CM', 'I_G', 'I_F', 'cendre', 'I1'])

        # Perform one-hot encoding
        new_inputs_encoded = perform_one_hot_encoding2(new_inputs_df)

        

        prediction = model_izod.predict(new_inputs_encoded)
        predicted_izod = prediction[0]  # Assign the predicted Izod value

    return render_template('izod.html', predicted_izod=predicted_izod)
    

@app.route('/melt', methods=['GET', 'POST'])
def melt():
    predicted_melt = None  # Initialize predicted Izod value as None
    if request.method == 'POST':
        # Retrieve form data
        items = request.form['items_m']
        couleur = request.form['couleur_m']
        I_CM = request.form['I_CM_m']
        I_G = request.form['I_G_m']
        I_F = request.form['I_F_m']
        densite = float(request.form['densite'])

        # Create a DataFrame for the selected items
        new_inputs_df = pd.DataFrame([[items, densite, couleur, I_CM, I_G, I_F]],
                                        columns=['Items', 'densite', 'COULEUR', 'I_CM', 'I_G', 'I_F'])
        

        # Perform one-hot encoding
        new_inputs_encoded = perform_one_hot_encoding_for_melt(new_inputs_df)


        # Make prediction
        predicted_melt = model_melt.predict(new_inputs_encoded)[0]

    return render_template('melt.html', predicted_melt=predicted_melt)
    
@app.route('/mixture_melt', methods=['GET', 'POST'])
def mixture_melt():
    prediction_mixture = None  # Initialize predicted Izod value as None

    if request.method == 'POST':
        component_count = int(request.form['component-count'])
        component_data = []
        input_data = pd.DataFrame(columns=['Items','Poids','Melt'])

        for i in range(component_count):
            item = request.form.get(f'item-{i}')
            weight = float(request.form.get(f'weight-{i}'))
            melt = float(request.form.get(f'melt-{i}'))
            component_data.append({'Items': item, 'Poids': weight, 'Melt': melt})

        # Create a DataFrame with the data
        input_data = pd.DataFrame(component_data)

        # Process the component data and generate predictions
        df_dataset = perform_melt_index_preprocessing(input_data)
        X = todataset(df_dataset,component_count)        
        X_melt_theorique = df_dataset['melt_theorique'].values
        X_pred = [X,X_melt_theorique]
        model = load_model("models\Models_Melange\model"+str(component_count)+".h5")
        prediction_mixture = pred(X_pred,model)
        prediction_mixture = prediction_mixture[0][0]
                 # Render the template with the prediction result
    return render_template('mixture_melt.html', prediction_mixture=prediction_mixture)


@app.route('/flexion', methods=['GET', 'POST'])
def flexion():
    predicted_flexion = None  # Initialize predicted flexion value as None

    if request.method == 'POST':
        items = request.form.get('items')
        melt = float(request.form.get('melt'))
        traction = float(request.form.get('traction'))
        couleur = request.form.get('couleur')
        i_cm = request.form.get('I_CM')
        i_g = request.form.get('I_G')
        i_f = request.form.get('I_F')
        cendre = float(request.form.get('cendre'))
        i1 = float(request.form.get('i1'))
 

        new_inputs_df = pd.DataFrame([[items, melt, traction, couleur, i_cm, i_g, i_f, cendre, i1]],
                                     columns=['Items', 'melt', 'traction', 'COULEUR', 'I_CM', 'I_G', 'I_F', 'cendre', 'I1',])
        model_flexion = pickle.load(open('models/rf_flexion.pkl', 'rb'))
        encoder_flex = pickle.load(open('models/encoder_flex.pkl', 'rb'))
        new_inputs_encoded = perform_one_hot_encoding(new_inputs_df,encoder_flex)
        model_flexion = pickle.load(open('models/rf_flexion.pkl', 'rb'))
        prediction = model_flexion.predict(new_inputs_encoded)  # Assuming you have a model for flexion
        predicted_flexion = prediction[0]

    return render_template('flexion.html', predicted_flexion=predicted_flexion)

@app.route('/traction', methods=['GET', 'POST'])
def traction():
    predicted_traction = None  # Initialize predicted flexion value as None

    if request.method == 'POST':
        items = request.form.get('items')
        melt = float(request.form.get('melt'))
        flexion = float(request.form.get('flexion'))
        couleur = request.form.get('couleur')
        i_cm = request.form.get('I_CM')
        i_g = request.form.get('I_G')
        i_f = request.form.get('I_F')
        cendre = float(request.form.get('cendre'))
        i1 = float(request.form.get('i1'))
 

        new_inputs_df = pd.DataFrame([[items, melt, flexion, couleur, i_cm, i_g, i_f, cendre, i1]],
                                     columns=['Items', 'melt', 'flex', 'COULEUR', 'I_CM', 'I_G', 'I_F', 'cendre', 'I1',])
        encoder_traction = pickle.load(open('models/encoder_traction.pkl', 'rb'))
        new_inputs_encoded = perform_one_hot_encoding(new_inputs_df,encoder_traction)
        model_traction = pickle.load(open('models/et_traction.pkl', 'rb'))
        prediction = model_traction.predict(new_inputs_encoded)  # Assuming you have a model for flexion
        predicted_traction= prediction[0]
        flash(f"Prediction: {predicted_traction:.2f}", 'success')


    return render_template('traction.html', predicted_traction=predicted_traction)

@app.route('/izod1', methods=['GET', 'POST'])
def izod1():
    predicted_izod1 = None  # Initialize predicted flexion value as None

    if request.method == 'POST':
        items = request.form.get('items')
        melt = float(request.form.get('melt'))
        flexion = float(request.form.get('flexion'))
        traction = float(request.form.get('traction'))
        couleur = request.form.get('couleur')
        i_cm = request.form.get('I_CM')
        i_g = request.form.get('I_G')
        i_f = request.form.get('I_F')
        cendre = float(request.form.get('cendre'))
 
        new_inputs_df = pd.DataFrame([[items, melt, flexion,traction, couleur, i_cm, i_g, i_f, cendre]],
                                     columns=['Items', 'melt', 'flex', 'traction','COULEUR', 'I_CM', 'I_G', 'I_F', 'cendre'])
        encoder_izod = pickle.load(open('models/encoder_izod1.pkl', 'rb'))
        new_inputs_encoded = perform_one_hot_encoding(new_inputs_df,encoder_izod)
        model_izod1 = pickle.load(open('models/et_I1.pkl', 'rb'))
        prediction = model_izod1.predict(new_inputs_encoded)  # Assuming you have a model for flexion
        predicted_izod1 = prediction[0]

    return render_template('izod1.html', predicted_izod1=predicted_izod1)

@app.route('/cendre', methods=['GET', 'POST'])
def cendre():
    predicted_cendre = None  # Initialize predicted flexion value as None

    if request.method == 'POST':
        items = request.form.get('items')
        melt = float(request.form.get('melt'))
        flexion = float(request.form.get('flexion'))
        traction = float(request.form.get('traction'))
        couleur = request.form.get('couleur')
        i_cm = request.form.get('I_CM')
        i_g = request.form.get('I_G')
        i_f = request.form.get('I_F')
        i1 = float(request.form.get('i1'))
 

        new_inputs_df = pd.DataFrame([[items, melt ,flexion,traction, couleur, i_cm, i_g, i_f,i1 ]],
                                     columns=['Items', 'melt','flex','traction', 'COULEUR', 'I_CM', 'I_G', 'I_F', 'I1'])
        encoder_cendre = pickle.load(open('models/encoder_cendre.pkl', 'rb'))
        new_inputs_encoded = perform_one_hot_encoding(new_inputs_df,encoder_cendre)
        model_cendre = pickle.load(open('models/et_cendre.pkl', 'rb'))
        prediction = model_cendre.predict(new_inputs_encoded)  # Assuming you have a model for flexion
        predicted_cendre = prediction[0]

    return render_template('cendre.html', predicted_cendre=predicted_cendre)

if __name__ == '__main__':
    app.run(debug=True)
