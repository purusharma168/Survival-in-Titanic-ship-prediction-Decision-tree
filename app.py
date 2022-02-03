from flask import Flask, escape, request, render_template
import pickle
import pandas as pd

model = pickle.load(open("DTCmodelForprediction.pkl", 'rb'))
scaler = pickle.load(open("StandardScaler.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                # print(dict_req)
                data = [list(dict_req.values())]
                print(data)
                # dat = pd.DataFrame(data)
                # print(dat.head())
                data_transform = scaler.fit_transform(data)
                print(data_transform)

                response = model.predict(data_transform)[0]
                print(response)

                if(response=='0'):
                    response="NO"
                else:
                    response="YES"
                return render_template("prediction.html", prediction_text="Chances of survived  - >"+str(response))

        except Exception as e:
            # print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}
            return render_template("prediction.html", prediction_text=error)



    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run(port=8000)