import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    modelnRF = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=4, max_features='auto',
                                     min_samples_leaf=2, min_samples_split=2, n_estimators=100)
    if request.method == 'POST':
        data = pd.read_pickle('linear_marking_dataset')
        X = data.drop('Sem 7 Pointer', axis=1)
        Y = data['Sem 7 Pointer']
        modelnRF.fit(X, Y)
        x = np.zeros(14)
        x[0] = request.form.get('ssc', type=float)
        x[1] = request.form.get('hsc', type=float)
        x[2] = request.form.get('sem1', type=float)
        x[3] = request.form.get('sem2', type=float)
        x[4] = request.form.get('sem3', type=float)
        x[5] = request.form.get('sem4', type=float)
        x[6] = request.form.get('sem5', type=float)
        x[7] = request.form.get('sem6', type=float)
        x[8] = request.form.get('avgsem', type=float)
        x[9] = int(request.form['pce'])
        x[10] = int(request.form['pse'])
        x[11] = int(request.form['resident'])
        x[12] = int(request.form['gender'])
        x[13] = int(request.form['distance'])
        # print(ssc, hsc, sem1, sem2, sem3, sem4, sem5, sem6, avgsem, pce, pse, resident, gender, distance)
        ans = modelnRF.predict([x])[0]
        b = float("{:.2f}".format(ans))
        print(b)
        if 9 <= b <= 10:
            grade = 'O'
        elif 8 <= b < 9:
            grade = 'A'
        elif 7 <= b < 8:
            grade = 'B'
        elif 6 <= b < 7:
            grade = 'C'
        elif 5 <= b < 6:
            grade = 'D'
        else:
            grade = 'E'

        return render_template('index2.html', res=b, grade=grade)


if __name__ == '__main__':
    app.run()
