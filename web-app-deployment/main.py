from flask import Flask, request, render_template
from tensorflow.keras import models, layers, backend as K
import joblib
import numpy as np

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

DeepNN = models.load_model("./models/DeepNN.h5", custom_objects={'F1': F1})
scaler = joblib.load('./files_for_training_model/scalerDeepNN.joblib')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        CreditScore = float(request.form.get("SelCreditScore"))
        Geography = request.form.get("SelGeography")
        Gender = request.form.get("SelGender")
        Age = float(request.form.get("SelAge"))
        Tenure = float(request.form.get("SelTenure"))
        Balance = float(request.form.get("SelBalance"))
        NumberOfProducts = float(request.form.get("NumberOfProducts"))
        HasCreditCard = request.form.get("HasCreditCard")
        IsActiveMember = request.form.get("IsActiveMember")
        EstimatedSalary = float(request.form.get("EstimatedSalary"))
                
        if Geography == "France":
            Germany = 0
            Spain = 0
        elif Geography == "Germany":
            Germany = 1
            Spain = 0
        else:
            Germany = 0
            Spain = 1

        if Gender == "Male":
            Male = 1
        else:
            Male = 0

        if HasCreditCard == "Yes_CC":
            HasCreditCard = 1
        else:
            HasCreditCard = 0

        if IsActiveMember == "Yes_AM":
            IsActiveMember = 1
        else:
            IsActiveMember = 0

        input_data = np.array([[CreditScore, Age, Tenure, Balance, NumberOfProducts, HasCreditCard,
        IsActiveMember, EstimatedSalary, Germany, Spain, Male]])

        std_data = scaler.transform(input_data)
        prediction = DeepNN.predict(std_data)
        prediction_label = np.round(prediction)

        if(prediction_label[0] == 0):
            answer = "Not Exited"
        else:
            answer = "Exited"

        bgcolor = "yellow"
        fontcolor = "red"

        return render_template("home.html", Result=answer, bgcolor=bgcolor, fontcolor=fontcolor)
    
    else:
        bgcolor = "#cccccc"
        fontcolor = "#666666"
        
    return render_template("home.html", bgcolor=bgcolor, fontcolor=fontcolor)


if __name__ == "__main__":
	app.run(debug=True)